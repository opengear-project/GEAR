import argparse
import json
import os
import time

import pandas as pd

# import tensor_parallel as tp
import torch
from tqdm import tqdm
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import transformers


TASKS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

choices = ["A", "B", "C", "D"]

def compute_metric(output_filename):
    with open(output_filename, "r") as f:
        run_results = json.load(f)
    total_acc = 0
    total_num = 0
    for task in run_results:
        acc = 0
        pred_answers = run_results[task]["pred_answers"]
        gold_answers = run_results[task]["gold_answers"]
        for pred, gold in zip(pred_answers, gold_answers):
            if pred == gold:
                acc += 1
        print("ACC-%s: %.4f" % (task, acc / len(gold_answers)))
        total_acc += acc
        total_num += len(gold_answers)
    print("ACC-all: %.4f" % (total_acc / total_num))


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


# def custom_stopping_criteria(input_ids, score, **kwargs):
#     stop_ids = [29871, 13, 13] # \n\n
#     return input_ids[-len(stop_ids)]


def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(
        prompts, return_tensors="pt", padding=True
    )
    input_tokens = {
        k: input_tokens[k] for k in input_tokens if k in ["input_ids", "attention_mask"]
    }
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to("cuda")

    return input_tokens

def load(args):
    model_kwargs = {}
    if "Llama-2" in args.model:
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
        model_kwargs["token"] = args.hf_token
  

    config = transformers.AutoConfig.from_pretrained(
        args.model,
        use_auth_token=True,
        token=args.hf_token,
        trust_remote_code=True,
    )

    from GEARLM import CompressionConfig, SimulatedGearLlamaForCausalLM

    compress_config = (
        None
        if args.compress_method == "None"
        else CompressionConfig(
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
    if "Llama" in args.model:
        model = SimulatedGearLlamaForCausalLM.from_pretrained(
            args.model,
            config=config,
            **model_kwargs,
            cache_dir="../cache",
            compress_config=compress_config,
        )
    elif "Mistral" in args.model:
        from GEARLM import SimulatedGearMistralForCausalLM

        model = SimulatedGearMistralForCausalLM.from_pretrained(
            args.model,
            config=config,
            **model_kwargs,
            cache_dir="../cache",
            compress_config=compress_config,
            trust_remote_code=True,
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model,
        token=args.hf_token,
        padding_side="left",
        model_max_length=args.model_max_length,
        cache_dir="../cache",
        trust_remote_code=True,
        # pad_token="<|endoftext|>",
    )
    if "Mistral" in args.model:
        tokenizer.pad_token = tokenizer.eos_token
    # model = model.to('cuda')
    return model, tokenizer