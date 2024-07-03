# evaluating CoT performance on MMLU
import os
import sys
import re
import time
import json
import argparse
import logging
import torch
import numpy as np

import datasets
import accelerate
import transformers
from tqdm.auto import tqdm
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, cast
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
from datasets import load_dataset
# from pastalib.pasta import PASTA

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class EvaluationSample:
    """Wrapper around format evaluation sample."""
    question: str 
    generation: str 
    emphasize_string: str
    target: str 
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


def extract_bbh_ans(ans, mode):
    ans = ans.split('\nQuestion:')[0].strip().strip(".")
    ans_line = ans.split('answer is ')
    # Expect to see 'answer is'. If not return whole string
    if len(ans_line) == 1:
        return ans
    else:
        ans = ans_line[-1].strip()
    
    if mode == 'multiple_choice':
        # options = ['A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.', 'H.', 'I.', 'J.', 'K.', 'L.', 'M.', 'N.', 'O.', 'P.', 'Q.', 'R.', 'S.', 'T.', 'U.', 'V.', 'W.', 'X.', 'Y.', 'Z.']
        options = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)', '(L)', '(M)', '(N)', '(O)', '(P)', '(Q)', '(R)', '(S)', '(T)', '(U)', '(V)', '(W)', '(X)', '(Y)', '(Z)']
        for option in options:
            if option in ans:
                # ans = option[0]
                ans = option
                break
        return ans
    elif mode == 'free_form':
        if ans[-1] == '.' and len(ans)>1:
            ans = ans[:-1]
        return ans

def extract_ans_mmlu(ans_model):
    ans_model = ans_model.split('\n')
    ans = []
    residual = []
    for li, al in enumerate(ans_model):
        ans.append(al)
        if('answer is' in al):
            break
    residual = list(ans_model[li + 1:])
    ans = '\n'.join(ans)
    residual = '\n'.join(residual)
    return ans, residual

def load_model_tokenizer(args):
    from GEARLM import CompressionConfig,SimulatedGearLlamaForCausalLM,SimulatedGearMistralForCausalLM
    model_kwargs = {}
    if any([name in args.model for name in ["Llama-2", "Vicuna", "Llama-3"]]):
        model_kwargs["torch_dtype"] = torch.float16 
        model_kwargs["device_map"] = "auto"
        model_kwargs["token"] = args.hf_token
        model_kwargs["cache_dir"] = "../cache"
    compress_config = (
        None
        if args.compress_method == "None"
        else CompressionConfig(
            compress_method=args.compress_method,
            rank=args.rank,
            rankv=args.rankv,
            prefill_rank = args.prefillrank,
            prefill_rankv = args.prefillrankv,
            
            loop=args.loop,
            quantize_bit=args.quantize_bit,
            group_num=args.group_num,
            group_size = args.group_size,
            top_k=args.top_kprun,
            left=args.left,
            attention_number=args.attention_number,
            device_num=args.gpu,
            batch_num=args.batch_size,

            streaming=args.streaming,
            streaming_gap=args.streaming_gap,
            stream_grouping=args.stream_grouping,
        )
    )
    if compress_config is not None:
        compress_config.copy_for_all_attention()
        compress_config.calculate_compress_ratio_list(4095, 4096)
    config = transformers.AutoConfig.from_pretrained(
        args.model, use_auth_token=True, token=args.hf_token, cache_dir = "../cache"
    )
    if "Llama" in args.model:
        model = SimulatedGearLlamaForCausalLM.from_pretrained(
                args.model,
                config=config,
                **model_kwargs,
                compress_config=compress_config,
            )
    elif "Mistral" in args.model:
        model = SimulatedGearMistralForCausalLM.from_pretrained(
                args.model,
                config=config,
                **model_kwargs,
                compress_config=compress_config,
            )
    else:
        raise ValueError(f"Unexpected model {args.model}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model,
        token=args.hf_token,
        padding_side="left",
        model_max_length=args.model_max_length,
        cache_dir = "../cache"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    model = model.to('cuda')
    return model, tokenizer

def load_head_config(head_config_input):
    if "{" in head_config_input and "}" in head_config_input:
        head_config = json.loads(head_config_input)
    elif ".json" in head_config_input:
        with open(head_config_input, "r") as handle:
            head_config = json.load(handle)
    elif "," in head_config_input or str(head_config_input).isdigit():
        head_config = [int(layer_idx) for layer_idx in head_config_input.split(',')]
    else:
        raise ValueError(f"Unexpected input {head_config_input}")
    return head_config


def prepare_bbh_example_prompt_with_cot(example, prompt_cot):
    question = example['input'] + '\n'
    for letter in ['A', 'B', 'C', 'D']:
        question += '(' + letter + ') ' + example[letter] + ' '
    question += "\nA: Let's think step by step."  
    prompt = prompt_cot + "\n\n" + question
    example['question'] = question 
    example['prompt'] = prompt
    return example

def prepare_bbh_example_prompt(examples):
    questions = []
    num_example = len(examples['input'])
    for idx in range(num_example):
        question = examples['input'][idx] + '\n'
        for letter in ['A', 'B', 'C', 'D']:
            question += '(' + letter + ') ' + examples[letter][idx] + ' '
        question += "\nA: Let's think step by step."  
        questions.append(question)
    examples['question'] = questions
    return examples

def prepare_bbh_question(question):
    options = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)', '(L)', '(M)', '(N)', '(O)', '(P)', '(Q)', '(R)', '(S)', '(T)', '(U)', '(V)', '(W)', '(X)', '(Y)', '(Z)']
    for option in options:
        question = question.replace(option, option[1]+'.')
    return question

def preprocess_aqua_dataset(dataset):
    def add_text_options(example):
        example["text_options"] = "\n".join(example["options"])
        return example
    processed_dataset = dataset.map(add_text_options)
    return processed_dataset

def prepare_aqua_prompts_and_targets(batch: dict, prompt_prefix:str=""):
    questions = batch["question"]
    options = batch["text_options"]
    prompts = [
        prompt_prefix + "\nQuestion: " + question + "\nOption:\n" + option + "\nThe answer is" \
            for question,option in zip(questions,options)
    ]
    targets = batch["correct"]
    return prompts, targets

def prepare_aqua_cot_prompts_and_targets(batch: dict, prompt_prefix:str=""):
    questions = batch["question"]
    options = batch["text_options"]
    prompts = [
        prompt_prefix + "\nQuestion: " + question + "\nOption:\n" + option + "\nLet's think step by step\n" \
            for question,option in zip(questions,options)
    ]
    targets = batch["correct"]
    return prompts, targets

def extract_aqua_answer(generation):
    # answer = generation.split('\nQuestion:')[0].strip().strip(".")
    if "answer is" in generation:
        generation = generation.split("answer is")[1]
    generation = generation.split("\nQuestion:")[0].strip()
    return generation

def evaluate_aqua_answer_zeroshot(generation, target, prompt):
    target_option = target + ")"
    target_answer = prompt.split("\nOption:")[1].split("\nThe answer is")[0].split(target+")")[-1].split("\n")[0]
    if generation.startswith("option") or generation.startswith("Option"):
        pred = generation[:15].split("\n")[0].strip()
        is_pred_true = target in pred
        return pred, target, is_pred_true
    elif any([generation.startswith(option) for option in ["A", "B", "C", "D", "E"]]):
        pred = generation[:15].split("\n")[0].strip() 
        is_pred_true = target in pred
        return pred, target, is_pred_true
    else:
        pred = extract_aqua_answer(generation)
        if target_answer in pred:
            return pred, target_answer, True
        else:
            is_pred_true = target_option in pred
            return pred, target_option, is_pred_true

def evaluate_aqua_answer_cot(generation, target, prompt):
    target_option = target + ")"
    target_answer = prompt.split("\nQuestion:")[-1].split("\nOption:")[1].split("\nLet's think step by step\n")[0].split(target+")")[-1].split("\n")[0]
    pred = extract_aqua_answer(generation)
    if target_answer in pred: 
        is_pred_true = (
            (target_answer in pred) and not all([option+")" in pred for option in ["A","B","C","D"]])
        )
        return pred, target_answer, is_pred_true
    else:
        is_pred_true = (
            (target_option in pred) and not all([option+")" in pred for option in ["A","B","C","D"]])
        )
        return pred, target_option, is_pred_true

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate AQuA Tasks")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b", help="Model name or path.")
    parser.add_argument("--apply_pasta", action="store_true", default=False, help="Apply PASTA.")
    parser.add_argument("--alpha", type=float, default=0.01, help="")
    parser.add_argument("--scale_position", type=str, default="exclude", help="")
    parser.add_argument("--emphasize_section", type=str, default=None, help="Where to emphasize with PASTA.")
    parser.add_argument("--head_config", type=str, default=None, help="PASTA head config for steering.")
    parser.add_argument("--update_layer", type=str, default=None, help="Layers to apply PASTA.")
    parser.add_argument("--occurrence", type=int, default=0, help="")

    parser.add_argument("--dataset", type=str, default="aqua_rat", help="Dataset from HF.")
    parser.add_argument("--prompt_file", type=str, default="lib_prompt/aqua/cot_prompt_8shots.txt", help="")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--max_length", type=int, default=None, help="")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="")
    parser.add_argument("--model_max_length", type=int, default=4096, help="")
    parser.add_argument("--do_sample", action="store_true", default=False, help="")
    parser.add_argument("--temperature", type=float, default=0.8, help="")
    parser.add_argument("--top_k", type=int, default=50, help="")
    parser.add_argument("--top_p", type=float, default=0.95, help="")
    parser.add_argument("--dataset_split", type=str, default="test", help="")
    parser.add_argument("--example_subset", type=str, default=None, help="")
    parser.add_argument("--hf_token", type=str, default=None, help="")
    parser.add_argument("--root_output_dir", type=str, default="outputs", help="Root output dir")
    parser.add_argument("--debug", action="store_true", default=False, help="")
    
    parser.add_argument("--compress_method", type=str, default="None", help="choose one of the compression method")
    parser.add_argument("--rank", type=float, default=0.0, help="rank compared with smaller dimension set to K cache.")
    parser.add_argument("--rankv", type=float, default=0.0, help="rank compared with smaller dimension set to V cache.")
    parser.add_argument("--prefillrank", type=float, default=0.0, help="rank compared with smaller dimension set to K cache.")
    parser.add_argument("--prefillrankv", type=float, default=0.0, help="rank compared with smaller dimension set to V cache.")
    parser.add_argument("--loop", type=int, default=0, help="loop of SVD solver, default = 0")
    parser.add_argument("--quantize_bit", type=int, default=8, help="Quantize bit of algorithm")
    parser.add_argument("--group_num", type=int, default=0, help="group number of group quantization")
    parser.add_argument("--group_size", type=int, default=0, help="")
    parser.add_argument("--top_kprun", type=float, default=0.0, help="")
    parser.add_argument("--left", type=float, default=0.0, help="outlier extraction part compared with total matrix")
    parser.add_argument("--attention_number", type=int, default=100, help="attention layer number of LLM, for LlAMA-2-7b it is 32")
    parser.add_argument("--gpu", type=int, default=0, help="")

    parser.add_argument("--streaming", action="store_true", default=False, help="Use streaming mode.")
    parser.add_argument("--streaming_gap", type=int, default=0, help="iteration length for re-compression")
    parser.add_argument("--stream_grouping", action="store_true", default=False, help="Use streaming mode.")

    torch.manual_seed(0)
    args = parser.parse_args()
    import random
    random.seed(0)
    logging.info("Loading Model and Tokenizer.")
    model, tokenizer = load_model_tokenizer(args)
    if args.debug:
        import ipdb
        ipdb.set_trace()

    if args.apply_pasta:
        if args.head_config is not None:
            head_config = load_head_config(args.head_config)
        pasta = PASTA(
            model, 
            tokenizer,
            head_config=head_config,
            alpha=args.alpha,
            scale_position=args.scale_position,
        )
    else:
        pasta = None

    root_output_dir = Path(args.root_output_dir)
    output_dir = "{pasta_config}_emphasize-{emphasize_section}".format(
        pasta_config = f"pasta-{args.alpha}-{args.scale_position}" if args.apply_pasta else "pasta-none",
        emphasize_section = f"{args.emphasize_section}",
    ) 
    if args.example_subset is not None:
        output_dir += f"_{args.dataset_split}-subset-{args.example_subset}"
    output_dir = (
        root_output_dir 
        # / f"{args.model.split('/')[-1]}" 
        / output_dir 
    )
    output_dir = output_dir / f"config-{str(args.head_config).split('/')[-1]}"
    output_dir.mkdir(exist_ok=True, parents=True)
    tb_writter = SummaryWriter(log_dir=str(output_dir.resolve()))
    logging.basicConfig(
        filename= os.path.join(str(output_dir.resolve()), 'log.txt'), filemode='a',
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)
    split = args.dataset_split if args.example_subset is None else f"{args.dataset_split}[{args.example_subset}]"

    total_acc = 0
    dataset = load_dataset(args.dataset, split=split)
    eval_dataset = preprocess_aqua_dataset(dataset)
    dataloader = torch.utils.data.DataLoader(
        cast(torch.utils.data.Dataset, eval_dataset),
        batch_size=args.batch_size,
    )
    results_file = output_dir / f"evaluation_{args.dataset_split}_result.json"

    all_samples = []
    # prompt_prefix = f""
    with open(args.prompt_file, "r") as f:
        prompt_prefix = f.read()
    for batch in tqdm(dataloader, desc=f"Evaluate {args.dataset_split}"):
        questions = batch["question"]
        prompts, targets = prepare_aqua_cot_prompts_and_targets(batch, prompt_prefix)
        
        if pasta is not None:
            inputs, offsets_mapping = pasta.inputs_from_batch(
                prompts, tokenizer, device="cuda"
            )
        else:
            tokenizer.pad_token = tokenizer.eos_token
            inputs = tokenizer(
                prompts, 
                return_tensors="pt",
                padding="longest",
                truncation=True,
            )
            inputs = inputs.to("cuda")
        print(inputs.input_ids.shape)
        generate_kwargs = dict(
            return_dict_in_generate=True,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        emphasize_substrings = [None]*len(prompts)
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

        if pasta is not None:
            if args.emphasize_section == "question":
                emphasize_substrings = questions
            elif args.emphasize_section == "generation":
                emphasize_substrings = [""]*len(prompts)
            elif args.emphasize_section == "prefix":
                emphasize_substrings = [prompt_prefix]*len(prompts)
            else: 
                raise ValueError(f"Unimplemented emphasize section: {args.emphasize_section}")
            with pasta.apply_steering(
                    model=model, 
                    strings=prompts, 
                    substrings=emphasize_substrings,
                    model_input=inputs, 
                    offsets_mapping=offsets_mapping,
                    occurrence=args.occurrence,
                ) as new_model:
                    outputs = new_model.generate(**inputs, **generate_kwargs)
        else:
            outputs = model.generate(**inputs, **generate_kwargs)
        generations = tokenizer.batch_decode(
            outputs.sequences[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        for question, generation, target, substring in zip(prompts, generations, targets, emphasize_substrings):
            pred, label, is_pred_true = evaluate_aqua_answer_cot(generation, target, question)
            if is_pred_true: 
                total_acc += 1

            sample = EvaluationSample(
                question=question,
                generation=generation,
                emphasize_string=substring,
                target=target, 
                pred=pred,
                label=label,
                is_pred_true=is_pred_true,
            )
            all_samples.append(sample)

    total_acc = total_acc / len(eval_dataset)
    evaluation_metric = EvaluationMetrics(accuracy=total_acc)
    evaluation_result = EvaluationResults(
        samples=all_samples, 
        metrics=evaluation_metric,
    )

    logger.info('Evaluate %s acc: %.4f' % (args.dataset_split, total_acc))
    tb_writter.add_scalar(f"{args.dataset_split}/accuracy", total_acc, 1)
    with results_file.open("w") as handle:
        json.dump(evaluation_result.to_dict(), handle)
