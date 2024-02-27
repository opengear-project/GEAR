import argparse
import json
import os
import logging
import re
import sys
import torch
import numpy as np
import datasets
import accelerate
import transformers

from tqdm.auto import tqdm
from pathlib import Path
from datasets import load_dataset
from typing import Any, Callable, Dict, Sequence, cast
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
from torch.utils.tensorboard import SummaryWriter
from models import LlamaForCausalLMNew
from transformers import LlamaForCausalLM
from models.utils import create_compress_config
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

MODEL_GENERATION_SPLIT = "\nQuestion: "
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationSample:
    """Wrapper around format evaluation sample."""

    question: str
    generation: str
    answer: str
    list_from_pred: list[str]
    list_from_answer: list[str]
    pred: float
    label: float
    is_pred_true: bool


@dataclass(frozen=True)
class EvaluationMetrics(DataClassJsonMixin):
    """Wrapper around aggregated evaluation metrics."""

    accuracy: float


@dataclass(frozen=True)
class EvaluationResults(DataClassJsonMixin):
    """Wrapper around evaluation results"""

    samples: list[EvaluationSample]
    metrics: EvaluationMetrics


def evaluate_pred_answer(pred_str, ans_str):
    pattern = "\d*\.?\d+"
    pred_str, ans_str = pred_str.replace(",", ""), ans_str.replace(",", "")
    pred_list = re.findall(pattern, pred_str)
    gold_list = re.findall(pattern, ans_str)
    if len(pred_list) >= 1:
        pred = float(pred_list[-1])
        gold = float(gold_list[-1])
        is_pred_true = pred == gold
    else:
        is_pred_true = False
        pred = None
        gold = float(gold_list[-1])
    return (
        is_pred_true,
        pred,
        pred_list,
        gold,
        gold_list,
    )


def test_answer(pred_str, ans_str):
    pattern = "\d*\.?\d+"
    pred = re.findall(pattern, pred_str)
    if len(pred) >= 1:
        print("#####\n Pred string:", pred_str, "\n pred_list", pred)
        pred = float(pred[-1].replace(",", ""))
        gold = re.findall(pattern, ans_str)
        print("\n Gold_answer", ans_str, "\n gold_list", gold)
        gold = float(gold[-1].replace(",", ""))
        print("\n result", gold, pred, gold == pred)
        return pred == gold
    else:
        return False


def parse_pred_ans(filename):
    with open(filename) as fd:
        lines = fd.readlines()
    am, a = None, None
    num_q, acc = 0, 0
    current_mode = "none"
    questions = []
    ans_pred = []
    ans_gold = []
    am_others = []
    for l in lines:
        if l.startswith("Q: "):
            if am is not None and a is not None:
                questions.append(q)
                ans_pred.append(am)
                ans_gold.append(a)
                if test_answer(am, a):
                    acc += 1
            current_mode = "q"
            q = l
            num_q += 1
        elif l.startswith("A_model:"):
            current_mode = "am"
            am = l
        elif l.startswith("A:"):
            current_mode = "a"
            a = l
        # TODO
        elif current_mode == "am" and l.startswith("Question: "):
            current_mode = "am_other"
            am_other = l
        else:
            if current_mode == "q":
                q += l
            elif current_mode == "am":
                am += l
            elif current_mode == "a":
                a += l
            elif current_mode == "am_other":
                am_other += l
            else:
                raise ValueError(current_mode)

    questions.append(q)
    ans_pred.append(am)
    ans_gold.append(a)
    am_others.append(am_other)
    if test_answer(am, a):
        acc += 1
    print("######\n num_q %d correct %d ratio %.4f" % (num_q, acc, float(acc / num_q)))
    return questions, ans_pred, ans_gold


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class StoppingCriteriaSub(transformers.StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if tokenizer.decode(stop) == tokenizer.decode(last_token):
                return True
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GSM8K Dataset")
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-2-7b", help="Model name or path."
    )
    parser.add_argument(
        "--prompt_file", type=str, default="prompt_original.txt", help=""
    )
    parser.add_argument("--hf_token", type=str, default=None, help="")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--example_subset", type=str, default=None, help="")
    parser.add_argument("--max_length", type=int, default=None, help="")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="")
    parser.add_argument("--model_max_length", type=int, default=4096, help="")
    parser.add_argument("--do_sample", action="store_true", default=False, help="")
    parser.add_argument("--temperature", type=float, default=0.8, help="")
    parser.add_argument("--top_k", type=int, default=50, help="")
    parser.add_argument("--top_p", type=float, default=0.95, help="")
    parser.add_argument(
        "--generation_split", type=str, default=MODEL_GENERATION_SPLIT, help=""
    )
    parser.add_argument(
        "--root_output_dir", type=str, default="outputs", help="Root output dir"
    )
    parser.add_argument("--debug", action="store_true", default=False, help="")
    parser.add_argument("--compress_method", type=str, default="None", help="")
    parser.add_argument("--rank", type=float, default=0.0, help="")
    parser.add_argument("--rankv", type=float, default=0.0, help="")
    parser.add_argument("--loop", type=int, default=0, help="")
    parser.add_argument("--quantize_bit", type=int, default=8, help="")
    parser.add_argument("--group_num", type=int, default=0, help="")
    parser.add_argument("--top_kprun", type=float, default=0.0, help="")
    parser.add_argument("--left", type=float, default=0.0, help="")
    parser.add_argument("--attention_number", type=int, default=100, help="")
    parser.add_argument("--stage", type=int, default=0, help="")
    parser.add_argument("--gpu", type=int, default=0, help="")
    parser.add_argument("--locality_saving", type=float, default=0.0, help="")
    parser.add_argument("--start_saving", type=float, default=0.0, help="")
    parser.add_argument(
        "--token_preserving", action="store_true", default=False, help=""
    )
    parser.add_argument("--heavy_size", type=int, default=0, help="")
    parser.add_argument("--recent_size", type=int, default=0, help="")
    parser.add_argument("--streaming", action="store_true", default=False, help="")
    parser.add_argument("--streaming_gap", type=int, default=0, help="")
    parser.add_argument("--zero_shot", action="store_true", default=False, help="")
    
    parser.add_argument("--weight-compress", type=str,choices=["uniform","GPTQ","AWQ"], default=False, help="")
    args = parser.parse_args()

    if args.debug:
        import ipdb

        ipdb.set_trace()

    # Setup output dir
    root_output_dir = Path(args.root_output_dir)
    output_dir = f"cot_{args.prompt_file.split('.')[0]}"
    if args.example_subset is not None:
        output_dir += f"_subset-{args.example_subset}"
    output_dir = root_output_dir / f"{args.model.split('/')[-1]}" / output_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    generation_file = (
        output_dir / f"generation_results_subset-{args.example_subset}.txt"
    )
    evaluation_result_file = output_dir / f"evaluation_gsm8k.json"

    split = "test" if args.example_subset is None else f"test[{args.example_subset}]"
    eval_dataset = load_dataset("gsm8k", "main", split=split)
    tb_writter = SummaryWriter(log_dir=str(output_dir.resolve()))
    logging.basicConfig(
        filename=os.path.join(output_dir.resolve(), "log.txt"),
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # Load Model and Tokenizer
    model_kwargs = {}
    logging.info("Loading Model and Tokenizer.")
    if "Llama-2" or "Qwen" or "Mistral" in args.model:
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
        model_kwargs["token"] = args.hf_token
        model_kwargs["cache_dir"] = "../cache"
        if args.weight_compress =="uniform":
            print("weight compress")
            model_kwargs["quantization_config"] = create_compress_config(
                None
            )
        elif args.weight_compress =="GPTQ":
            print("GPTQ")
            #change branch
            model_kwargs["revision"] = "gptq-8bit-128g-actorder_False"

    config = transformers.AutoConfig.from_pretrained(
        args.model, use_auth_token=True, token=args.hf_token, use_flash_attn=False,trust_remote_code=True
    )
    from transformers import LlamaTokenizer
    from models import GPT2CompressConfig

    compress_config = (
        None
        if args.compress_method == "None"
        else GPT2CompressConfig(
            compress_method=args.compress_method,
            rank=args.rank,
            rankv=args.rankv,
            loop=args.loop,
            quantize_bit=args.quantize_bit,
            group_num=args.group_num,
            top_k=args.top_kprun,
            left=args.left,
            attention_number=args.attention_number,
            device_num=args.gpu,
            batch_num=args.batch_size,
            stage=args.stage,
            start_saving=args.start_saving,
            locality_saving=args.locality_saving,
            token_preserving=args.token_preserving,
            heavy_size=args.heavy_size,
            recent_size=args.recent_size,
            streaming=args.streaming,
            streaming_gap=args.streaming_gap,
        )
    )
    if compress_config is not None:
        compress_config.copy_for_all_attention()
        compress_config.calculate_compress_ratio_list(4095, 4096)
    if "Llama-2" in args.model:
        if args.compress_method == "h2o":
            from models import H2OLlamaForCausalLM, LlamaConfig
            
            config = LlamaConfig.from_pretrained(
                args.model,
                use_auth_token=True,
                token=args.hf_token,
                use_flash_attn=False,
            )
            config.hh_size = 400
            config.recent_size = 50
            model = H2OLlamaForCausalLM.from_pretrained(
                args.model, config=config, **model_kwargs
            )
        else:
            model = LlamaForCausalLMNew.from_pretrained(
                args.model,
                config=config,
                **model_kwargs,
                compress_config=compress_config,
            )
        tokenizer = LlamaTokenizer.from_pretrained(
            args.model,
            token=args.hf_token,
            padding_side="left",
            model_max_length=args.model_max_length,
            use_fast=False,
            cache_dir="../cache",
        )
        tokenizer.pad_token = tokenizer.eos_token
    elif "Qwen" in args.model:
        from models import QWenLMHeadModel
        from transformers import AutoTokenizer

        model = QWenLMHeadModel.from_pretrained(
            args.model,
            config=config,
            **model_kwargs,
            compress_config=compress_config,
            trust_remote_code=True,
            # use_cahce = True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            token=args.hf_token,
            padding_side="left",
            model_max_length=args.model_max_length,
            use_fast=False,
            cache_dir="../cache",
            trust_remote_code=True,
            pad_token="<|endoftext|>",
            use_flash_attn=False,
        )
        
        # if "Qwen" in args.model:
        #     tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        #     tokenizer.pad_token = tokenizer.eos_token
    # model = model.to('cuda')
    elif "Mistral" in args.model:
        # from models import MixTralForCausalLM, MixtralConfig
        from transformers import AutoTokenizer
        # from transformers import MistralForCausalLM,MistralConfig
        from models import MistralForCausalLM,MistralConfig
        config = MistralConfig.from_pretrained(
            args.model,
            use_auth_token=True,
            token=args.hf_token,
            use_flash_attn=False,
            trust_remote_code=True,
        )
        model = MistralForCausalLM.from_pretrained(
            args.model,
            config=config,
            **model_kwargs,
            trust_remote_code=True,
            compress_config=compress_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            token=args.hf_token,
            padding_side="left",
            model_max_length=args.model_max_length,
            use_fast=False,
            cache_dir="../cache",
            trust_remote_code=True,
        )
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = tokenizer.eos_token
        pass
    logging.info("Preprocessing the dataset.")
    with open(f"lib_prompt/{args.prompt_file}", "r") as handle:
        prompt_cot = handle.read()
    # prompt_cot = open(f'lib_prompt/{args.prompt_file}').read()
    dataloader = torch.utils.data.DataLoader(
        cast(torch.utils.data.Dataset, eval_dataset),
        batch_size=args.batch_size,
    )

    all_samples = []
    all_question, all_generation, all_answer = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluate GSM8K"):
            questions = batch["question"]
            answers = batch["answer"]
            if args.zero_shot is True:
                prompt_cot = "answer the question through the form of The answer is xxx. Do not generate others."
                prompts = [
                    prompt_cot + "\nQuestion: " + question + "\n" for question in questions
                ]
            else:
                prompts = [
                    prompt_cot + "\nQuestion: " + question + "\n" for question in questions
                ]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
            )
            inputs = inputs.to("cuda")
            generate_kwargs = dict(
                return_dict_in_generate=True,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            if args.do_sample:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = args.temperature
                generate_kwargs["top_k"] = args.top_k
                generate_kwargs["top_p"] = args.top_p
            else:
                generate_kwargs["do_sample"] = False
                generate_kwargs["temperature"] = None
                generate_kwargs["top_k"] = None
                generate_kwargs["top_p"] = None
            outputs = model.generate(**inputs, **generate_kwargs)
            generations = tokenizer.batch_decode(
                outputs.sequences[:, inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )

            all_question += questions
            all_generation += generations
            all_answer += answers

            for question, generation, answer in zip(questions, generations, answers):
                is_pred_true, pred, pred_list, gold, gold_list = evaluate_pred_answer(
                    generation.split(args.generation_split)[0], answer
                )
                sample = EvaluationSample(
                    question=question,
                    generation=generation,
                    answer=answer,
                    list_from_pred=pred_list,
                    list_from_answer=gold_list,
                    pred=pred,
                    label=gold,
                    is_pred_true=is_pred_true,
                )
                all_samples.append(sample)

        accuracy = sum([sample.is_pred_true for sample in all_samples]) / len(all_samples)
        evaluation_metric = EvaluationMetrics(accuracy=accuracy)
        evaluation_result = EvaluationResults(
            samples=all_samples,
            metrics=evaluation_metric,
        )

    tb_writter.add_scalar("accuracy", accuracy, 1)
    logging.info(f"Accuracy: {accuracy}")

    with evaluation_result_file.open("w") as handle:
        json.dump(evaluation_result.to_dict(), handle)

    with generation_file.open("w") as handle:
        for question, generation, answer in zip(
            all_question, all_generation, all_answer
        ):
            handle.write(
                "Q: %s\nA_model:\n%s\nA:\n%s\n\n" % (question, generation, answer)
            )
