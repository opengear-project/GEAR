import logging
import math
import re
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from GEARLM import SimulatedGearLlamaForCausalLM, CompressionConfig

from datasets import load_dataset
from accelerate.utils import set_seed

from utils import (
    ModelArguments,
    DataArguments,
    CompressArguments,
    smart_tokenizer_and_embedding_resize,
)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
ANSWER_PROMPT = "The final answer is: "
QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"


def evaluation(model_args, data_args, compress_args):
    compress_config = (
        None
        if compress_args.compress_method == "None"
        else CompressionConfig(
            compress_method=compress_args.compress_method,
            rank=compress_args.rank,
            rankv=compress_args.rankv,
            loop=compress_args.loop,
            quantize_bit=compress_args.quantize_bit,
            group_num=compress_args.group_num,
            left=compress_args.left,
            attention_number=compress_args.attention_number,
            device_num=compress_args.gpu,
            batch_num=data_args.batch_size,
            start_saving=compress_args.start_saving,
            streaming=compress_args.streaming,
            streaming_gap=compress_args.streaming_gap,
        )
    )
    # TODO extend for all models not just 7B
    if compress_config is not None:
        compress_config.copy_for_all_attention()
        compress_config.calculate_compress_ratio_list(4095, 4096)
    model = SimulatedGearLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16,  # you may change it with different models
        token=model_args.token,
        device_map=model_args.device_map,
        cache_dir="../cache",
        compress_config=compress_config,
        use_cache=True,
    )
    model = model.to(compress_args.gpu)
    TOKEN_ID = "meta-llama/Llama-2-7b-hf"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        TOKEN_ID,
        token=model_args.token,
        model_max_length=model_args.model_max_length,
        padding_side="left",
        use_fast=False,
        cache_dir="../cache",
        trust_remote_code=True,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    logging.warning("Downloading Data")
    dataset = load_dataset(data_args.data_name, "main", cache_dir="../cache")
    test_set = dataset["test"]

    logging.warning("Formatting inputs...")
    question = [f"{example['question']}{QUESTION_PROMPT}" for example in test_set]
    answer = []

    # get numerical answer
    for example in test_set["answer"]:
        ans = example.split("####")[-1]
        ans = ans.replace(",", "")  # handle numbers like 2,000
        try:
            ans = float(ans)
        except ValueError:
            ans = float("inf")
        answer.append(ans)

    logging.warning("Tokenizing inputs...")
    eval_step = math.ceil(len(question) / data_args.batch_size)
    logging.warning(
        f"Total example: {len(question)} | eval batch size: {data_args.batch_size}"
        f"eval steps: {eval_step}"
    )
    question_data = []
    for i in range(eval_step):
        if i < eval_step - 1:
            batch = tokenizer(
                question[i * data_args.batch_size : (i + 1) * data_args.batch_size],
                return_tensors="pt",
                padding="longest",
            )
        else:
            batch = tokenizer(
                question[i * data_args.batch_size :],
                return_tensors="pt",
                padding="longest",
            )
        batch["input_len"] = len(batch["input_ids"][0])
        question_data.append(batch)

    model.eval()
    gen_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.8,
        "top_k": 40,
        "top_p": 0.95,
        "do_sample": True,
        "use_cache": True,
    }
    ans_pred_list = []
    set_seed(42)
    import time

    for step, batch in enumerate(question_data):
        with torch.no_grad():
            if "Qwen" not in model_args.model_name_or_path:
                gen_kwargs["input_ids"] = batch["input_ids"].to(compress_args.gpu)
                gen_kwargs["attention_mask"] = batch["attention_mask"].to(
                    compress_args.gpu
                )
                start = time.time()
                generated_tokens = model.generate(**gen_kwargs)
            else:
                start = time.time()
                generated_tokens = model.generate(
                    input_ids=batch["input_ids"].to(compress_args.gpu)
                )
            torch.cuda.synchronize()
            end = time.time()

            print(end - start)

        pred_tokens = generated_tokens[:, batch["input_len"] :]
        decoded_pred = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)

        # Extract the numbers in sentences
        print(decoded_pred)
        ans_pred_list += [
            extract_answer_number(sentence_pred) for sentence_pred in decoded_pred
        ]
        accuracy = compute_accuracy(answer, ans_pred_list)
        print("accuracy", accuracy)

    print("prediction", ans_pred_list)
    print("ground truth", answer)

    accuracy = compute_accuracy(answer, ans_pred_list)

    print(
        f"adapter: {model_args.adapter_name_or_path} | GSM8K test accuracy: {accuracy} | "
        f"full precision: {model_args.full_precision}"
    )


def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", sentence)]
    if not pred:
        return float("inf")
    segment = sentence.split(ANSWER_PROMPT)
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r"-?\d+\.?\d*", pred_answer)]
        if len(pred_answer) > 0:
            pred_answer = pred_answer[0]
        else:
            pred_answer = float(pred[-1])
    else:
        # use the last number as the answer
        pred_answer = float(pred[-1])

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float("inf")
    return pred_answer


def compute_accuracy(pred: list, gold: list):
    acc = 0.0
    for p, g in zip(pred, gold):
        if p == g:
            acc += 1

    return acc / len(pred)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, CompressArguments)
    )
    model_args, data_args, compress_args = parser.parse_args_into_dataclasses()
    if model_args.ckpt_dir is not None:
        adapter_dir_list = [
            os.path.join(model_args.ckpt_dir, ckpt_dir)
            for ckpt_dir in os.listdir(model_args.ckpt_dir)
            if "checkpoint-" in ckpt_dir
        ]
    elif model_args.adapter_name_or_path is not None:
        adapter_dir_list = [model_args.adapter_name_or_path]
    else:
        logging.warning(
            "Use the checkpoint in HF hub, stored in the `subfolder='gsm8k'` in target model."
        )
        adapter_dir_list = [None]

    for adapter_path in adapter_dir_list:
        model_args.adapter_name_or_path = adapter_path
        evaluation(model_args, data_args, compress_args)
