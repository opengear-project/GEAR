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
from torch.utils.tensorboard import SummaryWriter

from models.utils import create_compress_config
logger = logging.getLogger(__name__)

MULTIPLE_CHOICE_TASKS = [
    'temporal_sequences', 
    'disambiguation_qa', 
    'date_understanding', 
    'tracking_shuffled_objects_three_objects', 
    'penguins_in_a_table', 
    'geometric_shapes', 
    'snarks', 
    'ruin_names', 
    'tracking_shuffled_objects_seven_objects', 
    'tracking_shuffled_objects_five_objects', 
    'logical_deduction_three_objects', 
    'hyperbaton', 
    'logical_deduction_five_objects', 
    'logical_deduction_seven_objects', 
    'movie_recommendation', 
    'salient_translation_error_detection', 
    'reasoning_about_colored_objects', 
]
FREE_FORM_TASKS = [
    'multistep_arithmetic_two', 
    'navigate', 
    'dyck_languages', 
    'word_sorting', 
    'sports_understanding', 
    'boolean_expressions', 
    'object_counting', 
    'formal_fallacies', 
    'causal_judgement', 
    'web_of_lies', 
]

TASKS = MULTIPLE_CHOICE_TASKS + FREE_FORM_TASKS

@dataclass(frozen=True)
class EvaluationSample:
    """Wrapper around format evaluation sample."""
    question: str 
    generation: str 
    target: str 
    extract_ans: str
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


def extract_ans(ans, mode):
    if not ans.startswith("\n\nQ: ") and "\n\nQ: " in ans:
        ans = ans.split("\n\nQ: ")[0]
    ans_line = ans.split('answer is ')
    # Expect to see 'answer is'. If not return whole string
    if len(ans_line) == 1:
        return ans
    else:
        ans = ans_line[-1].strip()
    
    if mode == 'multiple_choice':
        if len(ans)>3 and ans[0] == "(" and ans[2] == ")":
            return ans[1]
        options = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)', '(L)', '(M)', '(N)', '(O)', '(P)', '(Q)', '(R)', '(S)', '(T)', '(U)', '(V)', '(W)', '(X)', '(Y)', '(Z)']
        for option in options:
            if option in ans:
                ans = option[1]
                break
        return ans
    elif mode == 'free_form':
        if len(ans)>0 and ans[-1] == '.':
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

def test_answer_mmlu_(pred_str, ans):
    pattern = 'the answer is ('
    pred = pred_str.lower().split(pattern)
    
    if(len(pred) > 1 and len(pred[1])>0):
        # print(pred)
        pred = pred[1][0]
        gold = ans.lower()
        # print('debug 1, pred %s, gold %s' % (pred, gold))
        return pred == gold, pred, gold
    else: 
        pred = 'NA'
        # print(ans_str)
        gold = ans.lower()
        # print('debug 2, pred %s, gold %s' % (pred, gold))
        return pred == gold, pred, gold

def load_model_tokenizer(args):
    model_kwargs = {}
    from models import LlamaForCausalLMNew,QWenLMHeadModel, GPT2CompressConfig
    # add compression compression config
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
            streaming=args.streaming,
            streaming_gap=args.streaming_gap,
        )
    )
    if compress_config is not None:
        compress_config.copy_for_all_attention()
        compress_config.calculate_compress_ratio_list(4095, 4096)
    if "Llama-2" or "Mistral" in args.model:
        model_kwargs["torch_dtype"] = torch.float16 
        model_kwargs["device_map"] = "auto"
        model_kwargs["token"] = args.hf_token
        model_kwargs["cache_dir"] = "../cache"
        if args.weight_compress:
            print("weight compress")
            model_kwargs["quantization_config"] = create_compress_config(
                None
            )
    config = transformers.AutoConfig.from_pretrained(
        args.model, use_auth_token=True, token=args.hf_token,
    )
    if "Llama" in args.model:
        model = LlamaForCausalLMNew.from_pretrained(
            args.model, config=config, **model_kwargs,compress_config=compress_config,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model,
            token=args.hf_token,
            padding_side="left",
            model_max_length=args.model_max_length,
            cache_dir="../cache",
        )
    else:
        # mistral
        from models import MistralForCausalLM
        model = MistralForCausalLM.from_pretrained(
            args.model, config=config, **model_kwargs,compress_config=compress_config,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model,
            token=args.hf_token,
            padding_side="left",
            model_max_length=args.model_max_length,
            cache_dir="../cache",
        )
    tokenizer.pad_token = tokenizer.eos_token
    # model = model.to('cuda')
    return model, tokenizer

def prepare_prompt_example_with_cot(example, prompt_cot):
    question = example['input'] + '\n'
    for letter in ['A', 'B', 'C', 'D']:
        question += '(' + letter + ') ' + example[letter] + ' '
    question += "\nA: Let's think step by step."  
    prompt = prompt_cot + "\n\n" + question
    example['question'] = question 
    example['prompt'] = prompt
    return example

def prepare_example_prompt(examples):
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

def get_method_name(args):
    if args.compress_method == "uniformquantization":
        return f"uniformquant" 
    elif args.compress_method == "outquantize_with_lrap":
        return f"lrapoutquant_rank-{args.rank}-{args.rankv}_left-{args.left}"
    elif args.compress_method == "densesparseuniformquantization":
        return f"denspaquant-{args.left}"
    elif args.compress_method == "groupquantization":
        return f"groupquant"
    else:
        return f"base"


def main(args):
    logging.info("Loading Model and Tokenizer.")
    model, tokenizer = load_model_tokenizer(args)
    tasks = args.tasks

    root_output_dir = Path(args.root_output_dir)
    output_dir = "{prompt}_{name}_bit-{bit}_attnum-{attnum}_len-{max_new_tokens}_stream-{streaming_gap}".format(
        prompt="cot" if not args.zeroshot else "zeroshot",
        name=get_method_name(args),
        bit=args.quantize_bit, 
        attnum=args.attention_number, 
        max_new_tokens=args.max_new_tokens,
        streaming_gap=args.streaming_gap, 
    )
    if args.example_subset is not None:
        output_dir += f"_subset-{args.example_subset}"
    output_dir = (
        root_output_dir 
        / f"{args.model.split('/')[-1]}" 
        / output_dir
    )
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

    # mmlu_prompt = json.load(open(args.prompt_file))
    split = args.dataset_split if args.example_subset is None else f"{args.dataset_split}[{args.example_subset}]"

    all_ave_acc, num_task = 0, 0 
    for task in tasks:
        acc = 0
        eval_dataset = load_dataset("lukaemon/bbh", task, split=split,cache_dir="../cache")
        # eval_dataset = eval_dataset.map(
        #     prepare_example_prompt, batched=True, desc="Prepare question", 
        # )
        dataloader = torch.utils.data.DataLoader(
            cast(torch.utils.data.Dataset, eval_dataset),
            batch_size=args.batch_size,
        )
        logger.info('Testing %s ...' % task)
        generation_file = output_dir / f"{task}.txt"
        results_file = output_dir / f"{task}_result.json"

        all_samples = []
        # prompt_cot = mmlu_prompt[task]
        if not args.zeroshot: 
            instruction_prompt = open('lib_prompt/%s.txt' % task, 'r').read()
        else:
            instruction_prompt = "\nAnswer the following question.\n"
        with generation_file.open("w") as fd:
            for batch in tqdm(dataloader, desc=f"Evaluate {task}"):
                questions = batch["input"]
                if not args.zeroshot:
                    prompts = [
                        instruction_prompt+"\n\nQ: "+question+"\nA: Let's think step by step." for question in questions
                    ]
                else:
                    beg_answer_prompt = "\nA: the answer is" if task in MULTIPLE_CHOICE_TASKS else ""
                    prompts = [
                        instruction_prompt+"\n\nQ: "+question+beg_answer_prompt for question in questions
                    ]
                targets = batch["target"]

                inputs = tokenizer(
                    prompts, # [prompt_q],
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

                for question, ans_model, target in zip(questions, generations, targets):
                    mode = "multiple_choice" if task in MULTIPLE_CHOICE_TASKS else "free_form"
                    ans_ = extract_ans(ans_model, mode)
                    if mode == 'multiple_choice':
                        gold = target[1]
                    else:
                        gold = target
                    pred = ans_
                    is_pred_true = (pred==gold)
                    # is_pred_true, pred, gold = test_answer_mmlu_(ans_, target)
                    if(is_pred_true): 
                        acc += 1

                    sample = EvaluationSample(
                        question=question,
                        generation=ans_model,
                        target=target, 
                        extract_ans=ans_, 
                        pred=pred,
                        label=gold,
                        is_pred_true=is_pred_true,
                    )
                    all_samples.append(sample)
                    fd.write('Q: %s\nA_model:\n%s\nA_target:\n%s\n\n' % (question, ans_, target))

            task_acc = acc / len(eval_dataset)
            evaluation_metric = EvaluationMetrics(accuracy=task_acc)
            evaluation_result = EvaluationResults(
                samples=all_samples, 
                metrics=evaluation_metric,
            )
        
        all_ave_acc += task_acc
        num_task += 1 

        logger.info('%s acc %.4f' % (task, task_acc))
        tb_writter.add_scalar(f"{task}/accuracy", task_acc, 1)
        with results_file.open("w") as handle:
            json.dump(evaluation_result.to_dict(), handle)
        
    all_ave_acc = all_ave_acc/num_task
    logger.info('Average Acc: %.4f' % (all_ave_acc))
    tb_writter.add_scalar("Average Acc", all_ave_acc, 1)
    return all_ave_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate MMLU Tasks"
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b", help="Model name or path.")
    parser.add_argument(
        "--tasks", nargs="+", type=str, default=TASKS, 
        help="The evaluation tasks."
    )
    parser.add_argument("--prompt_file", type=str, default="lib_prompt/mmlu-cot.json", help="")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--max_length", type=int, default=None, help="")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="")
    parser.add_argument("--model_max_length", type=int, default=3072, help="")
    parser.add_argument("--do_sample", action="store_true", default=False, help="")
    parser.add_argument("--temperature", type=float, default=0.8, help="")
    parser.add_argument("--top_k", type=int, default=50, help="")
    parser.add_argument("--top_p", type=float, default=0.95, help="")
    parser.add_argument("--zeroshot", action="store_true", default=False, help="")
    parser.add_argument("--dataset_split", type=str, default="test", help="")
    parser.add_argument("--example_subset", type=str, default=None, help="")
    parser.add_argument("--hf_token", type=str, default=None, help="")
    parser.add_argument("--root_output_dir", type=str, default="outputs", help="Root output dir")
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
    parser.add_argument("--streaming", action="store_true", default=False, help="")
    parser.add_argument("--streaming_gap", type=int, default=0, help="")
    parser.add_argument("--weight-compress", action="store_true", default=False, help="")
    args = parser.parse_args()
    if args.debug:
        import ipdb
        ipdb.set_trace()
    
    main(args)