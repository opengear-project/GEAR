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

logger = logging.getLogger(__name__)

TASKS = [
        'abstract_algebra',
        'anatomy',
        'astronomy',
        'business_ethics',
        'clinical_knowledge',
        'college_biology',
        'college_chemistry',
        'college_computer_science',
        'college_mathematics',
        'college_medicine',
        'college_physics',
        'computer_security',
        'conceptual_physics',
        'econometrics',
        'electrical_engineering',
        'elementary_mathematics',
        'formal_logic',
        'global_facts',
        'high_school_biology',
        'high_school_chemistry',
        'high_school_computer_science',
        'high_school_european_history',
        'high_school_geography',
        'high_school_government_and_politics',
        'high_school_macroeconomics',
        'high_school_mathematics',
        'high_school_microeconomics',
        'high_school_physics',
        'high_school_psychology',
        'high_school_statistics',
        'high_school_us_history',
        'high_school_world_history',
        'human_aging',
        'human_sexuality',
        'international_law',
        'jurisprudence',
        'logical_fallacies',
        'machine_learning',
        'management',
        'marketing',
        'medical_genetics',
        'miscellaneous',
        'moral_disputes',
        'moral_scenarios',
        'nutrition',
        'philosophy',
        'prehistory',
        'professional_accounting',
        'professional_law',
        'professional_medicine',
        'professional_psychology',
        'public_relations',
        'security_studies',
        'sociology',
        'us_foreign_policy',
        'virology',
        'world_religions']

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


def extract_ans(ans_model):
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
    if "Llama-2" in args.model:
        model_kwargs["torch_dtype"] = torch.float16 
        model_kwargs["device_map"] = "auto"
        model_kwargs["token"] = args.hf_token
    
    config = transformers.AutoConfig.from_pretrained(
        args.model, use_auth_token=True, token=args.hf_token,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, config=config, **model_kwargs
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model,
        token=args.hf_token,
        padding_side="left",
        model_max_length=args.model_max_length,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = model.to('cuda')
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


def main(args):
    logging.info("Loading Model and Tokenizer.")
    model, tokenizer = load_model_tokenizer(args)
    tasks = args.tasks
    if args.debug:
        import ipdb
        ipdb.set_trace()

    root_output_dir = Path(args.root_output_dir)
    output_dir = "cot_base"
    if args.example_subset is not None:
        output_dir += f"_subset-{args.example_subset}"
    output_dir = (
        root_output_dir 
        / f"{args.model.split('/')[-1]}_{args.prompt_file.split('/')[-1].split('.')[0]}" 
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

    mmlu_prompt = json.load(open(args.prompt_file))
    split = args.dataset_split if args.example_subset is None else f"{args.dataset_split}[{args.example_subset}]"

    all_ave_acc, num_task = 0, 0 
    for task in tasks:
        acc = 0
        eval_dataset = load_dataset("lukaemon/mmlu", task, split=split)
        eval_dataset = eval_dataset.map(
            prepare_example_prompt, batched=True, desc="Prepare question", 
        )
        dataloader = torch.utils.data.DataLoader(
            cast(torch.utils.data.Dataset, eval_dataset),
            batch_size=args.batch_size,
        )
        logger.info('Testing %s ...' % task)
        generation_file = output_dir / f"{task}.txt"
        results_file = output_dir / f"{task}_result.json"

        all_samples = []
        prompt_cot = mmlu_prompt[task]
        with generation_file.open("w") as fd:
            for batch in tqdm(dataloader, desc=f"Evaluate {task}"):
                questions = batch["question"]
                prompts = [prompt_cot+"\n\n"+question for question in questions]
                targets = batch["target"]

            # for q_ in tqdm(eval_dataset, desc=f"Evaluate {task}"):
            #     q = q_['input'] + '\n'
            #     for letter in ['A', 'B', 'C', 'D']:
            #         q += '(' + letter + ') ' + q_[letter] + ' '
            #     q += "\nA: Let's think step by step."  
            #     prompt_q = mmlu_prompt[task] + "\n\n" + q

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
                    ans_, residual = extract_ans(ans_model)
                    is_pred_true, pred, gold = test_answer_mmlu_(ans_, target)
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
                    fd.write('Q: %s\nA_model:\n%s\nA:\n%s\n\n' % (question, ans_, target))

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

    args = parser.parse_args()
    main(args)