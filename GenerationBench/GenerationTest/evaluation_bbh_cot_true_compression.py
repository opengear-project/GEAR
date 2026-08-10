"""
BBH-CoT evaluation for the TrueCompression method (GearLlamaForCausalLMNew / GearMistralForCausalLMNew).

NOTE: TrueCompression supports Llama and Mistral-architecture models.
      Pass a local or HF Llama or Mistral checkpoint with --model.
      The script automatically detects the model architecture.

This script is based on evaluation_bbh_cot.py and evaluation_gsm8k_true_compression.py:
  - Imports GearLlamaForCausalLMNew for Llama or GearMistralForCausalLMNew for Mistral
  - Builds compress_config as a plain dict (no copy_for_all_attention / calculate_compress_ratio_list)
  - Adds --stream, --buffer_len, --compress_mode args for the new cache path
  - Auto-selects device: cuda → mps → cpu
  - Auto-detects model architecture (Llama or Mistral) and uses appropriate model class
  - Writes output JSON to evaluation_bbh_cot_true_compression.json to avoid clobbering Simulated results
"""

import os
import sys
import re
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
from GEARLM import GearLlamaForCausalLMNew
from GEARLM.TrueCompression.models.TrueCompressionMistral import GearMistralForCausalLMNew
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

MULTIPLE_CHOICE_TASKS = [
    # "temporal_sequences",
    # "disambiguation_qa",
    # "date_understanding",
    # "tracking_shuffled_objects_three_objects",
    # "penguins_in_a_table",
    # "geometric_shapes",
    # "snarks",
    # "ruin_names",
    # "tracking_shuffled_objects_seven_objects",
    # "tracking_shuffled_objects_five_objects",
    # "logical_deduction_three_objects",
    # "hyperbaton",
    # "logical_deduction_five_objects",
    # "logical_deduction_seven_objects",
    # "movie_recommendation",
    # "salient_translation_error_detection",
    "reasoning_about_colored_objects",
]
FREE_FORM_TASKS = [
    "multistep_arithmetic_two",
    "navigate",
    "dyck_languages",
    "word_sorting",
    "sports_understanding",
    "boolean_expressions",
    "object_counting",
    "formal_fallacies",
    "causal_judgement",
    "web_of_lies",
]

TASKS = MULTIPLE_CHOICE_TASKS + FREE_FORM_TASKS


@dataclass(frozen=True)
class EvaluationSample:
    """Wrapper around format evaluation sample."""
    question: str
    generation: str
    target: str
    extract_ans: str
    pred: str
    label: str
    is_pred_true: bool


@dataclass(frozen=True)
class EvaluationMetrics(DataClassJsonMixin):
    """Wrapper around aggregated evaluation metrics."""
    accuracy: float


@dataclass(frozen=True)
class EvaluationResults(DataClassJsonMixin):
    """Wrapper around evaluation results"""
    samples: list
    metrics: EvaluationMetrics


def extract_ans(ans, mode):
    if not ans.startswith("\n\nQ: ") and "\n\nQ: " in ans:
        ans = ans.split("\n\nQ: ")[0]
    ans_line = ans.split("answer is ")
    # Expect to see 'answer is'. If not return whole string
    if len(ans_line) == 1:
        return ans
    else:
        ans = ans_line[-1].strip()

    if mode == "multiple_choice":
        if len(ans) > 3 and ans[0] == "(" and ans[2] == ")":
            return ans[1]
        options = [
            "(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)",
            "(I)", "(J)", "(K)", "(L)", "(M)", "(N)", "(O)", "(P)",
            "(Q)", "(R)", "(S)", "(T)", "(U)", "(V)", "(W)", "(X)",
            "(Y)", "(Z)",
        ]
        for option in options:
            if option in ans:
                ans = option[1]
                break
        return ans
    elif mode == "free_form":
        if len(ans) > 0 and ans[-1] == ".":
            ans = ans[:-1]
        return ans


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BBH-CoT Tasks – TrueCompression")
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Path or HF hub name of a Llama-architecture checkpoint. "
             "Defaults to TinyLlama for MPS smoke tests.",
    )
    parser.add_argument(
        "--tasks", nargs="+", type=str, default=TASKS, help="The evaluation tasks."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--max_length", type=int, default=None, help="max length")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="max generation length")
    parser.add_argument("--model_max_length", type=int, default=4096, help="model max input length")
    parser.add_argument("--do_sample", action="store_true", default=False, help="argument for generation")
    parser.add_argument("--temperature", type=float, default=0.8, help="argument for generation")
    parser.add_argument("--top_k", type=int, default=50, help="argument for generation")
    parser.add_argument("--top_p", type=float, default=0.95, help="argument for generation")
    parser.add_argument("--zeroshot", action="store_true", default=False, help="whether use zeroshot or cot")
    parser.add_argument("--dataset_split", type=str, default="test", help="which part of dataset to choose")
    parser.add_argument("--example_subset", type=str, default=None, help="which part of dataset to choose")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token")
    parser.add_argument(
        "--root_output_dir", type=str, default="outputs", help="Root output dir"
    )
    parser.add_argument("--debug", action="store_true", default=False, help="")
    # TrueCompression-specific args
    parser.add_argument(
        "--compress_method",
        type=str,
        default="None",
        help="'None' to disable compression, or e.g. 'gear', 'uniform', 'outlier'.",
    )
    parser.add_argument(
        "--compress_mode",
        type=str,
        default="gear",
        help="Compress mode passed to CompressedUnion (gear / uniform / outlier / …).",
    )
    parser.add_argument("--quantize_bit", type=int, default=8, help="")
    parser.add_argument("--rank", type=float, default=0.0, help="")
    parser.add_argument("--loop", type=int, default=0, help="")
    parser.add_argument("--left", type=float, default=0.0, help="")
    parser.add_argument(
        "--buffer_len",
        type=int,
        default=20,
        help="Tail buffer length before compress() flushes (for new method).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Use StreamCompressedCache instead of CompressedCache.",
    )
    parser.add_argument(
        "--streaming_gap",
        type=int,
        default=1,
        help="Recompress interval for streaming cache.",
    )
    parser.add_argument("--sink_tokens", type=int, default=4,
        help="Number of initial prefill tokens always kept in full precision.")
    parser.add_argument("--recency_tokens", type=int, default=64,
        help="Size of the rolling recency FP window.")

    args = parser.parse_args()

    if args.debug:
        import ipdb
        ipdb.set_trace()

    # Output paths
    root_output_dir = Path(args.root_output_dir)
    output_dir = "cot_bbh_true_compression" if not args.zeroshot else "zeroshot_bbh_true_compression"
    if args.example_subset is not None:
        output_dir += f"_subset-{args.example_subset}"
    output_dir = root_output_dir / f"{args.model.split('/')[-1]}" / output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    # Logging
    tb_writter = SummaryWriter(log_dir=str(output_dir.resolve()))
    logging.basicConfig(
        filename=os.path.join(str(output_dir.resolve()), "log.txt"),
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    # Device selection: cuda > mps > cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # Build compress_config dict (for new method)
    if args.compress_method == "None":
        compress_config = None
    else:
        compress_config = {
            "compress_mode": args.compress_mode,
            "quantize_bit": args.quantize_bit,
            "rank": args.rank,
            "loop": args.loop,
            "left": args.left,
            "buffer_len": args.buffer_len,
            "stream": args.stream,
            "streaming_gap": args.streaming_gap,
            "sink_tokens": args.sink_tokens,
            "recency_tokens": args.recency_tokens,
        }
    logging.info(f"compress_config: {compress_config}")

    # Model + tokenizer
    if device.type == "cuda":
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "cache_dir": "../cache",
        }
    else:
        model_kwargs = {
            "torch_dtype": torch.float16,
            "cache_dir": "../cache",
        }
    if args.hf_token is not None:
        model_kwargs["token"] = args.hf_token

    config = transformers.AutoConfig.from_pretrained(
        args.model,
        use_flash_attn=False,
        trust_remote_code=True,
        **({"token": args.hf_token} if args.hf_token else {}),
    )

    # Detect model architecture and select appropriate model class
    model_type = config.model_type.lower()
    if "mistral" in model_type:
        ModelClass = GearMistralForCausalLMNew
        model_arch_name = "Mistral"
    elif "llama" in model_type:
        ModelClass = GearLlamaForCausalLMNew
        model_arch_name = "Llama"
    else:
        raise ValueError(
            f"Unsupported model architecture: {model_type}. "
            f"Supported architectures: llama, mistral"
        )

    logging.info(f"Loading TrueCompression {model_arch_name} model from: {args.model}")
    model = ModelClass.from_pretrained(
        args.model,
        config=config,
        compress_config=compress_config,
        **model_kwargs,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    _p0 = next(model.parameters())
    logging.info(
        "Model parameters on device: %s (sample dtype=%s)",
        _p0.device,
        _p0.dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        padding_side="left",
        model_max_length=args.model_max_length,
        use_fast=False,
        cache_dir="../cache",
        **({"token": args.hf_token} if args.hf_token else {}),
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Evaluation loop
    tasks = args.tasks
    all_ave_acc, num_task = 0, 0

    for task in tasks:
        acc = 0
        eval_dataset = load_dataset(
            "lukaemon/bbh", task, split=args.dataset_split, cache_dir="../cache"
        )
        dataloader = torch.utils.data.DataLoader(
            cast(torch.utils.data.Dataset, eval_dataset),
            batch_size=args.batch_size,
        )
        logger.info("Testing %s ..." % task)
        generation_file = output_dir / f"{task}.txt"
        results_file = output_dir / f"{task}_result.json"

        all_samples = []

        if not args.zeroshot:
            instruction_prompt = open("lib_prompt/%s.txt" % task, "r").read()
        else:
            instruction_prompt = "\nAnswer the following question.\n"

        with torch.no_grad():
            with generation_file.open("w", encoding="utf-8") as fd:
                for batch in tqdm(dataloader, desc=f"Evaluate {task}"):
                    questions = batch["input"]
                    if not args.zeroshot:
                        prompts = [
                            instruction_prompt
                            + "\n\nQ: "
                            + question
                            + "\nA: Let's think step by step."
                            for question in questions
                        ]
                    else:
                        beg_answer_prompt = (
                            "\nA: the answer is" if task in MULTIPLE_CHOICE_TASKS else ""
                        )
                        prompts = [
                            instruction_prompt + "\n\nQ: " + question + beg_answer_prompt
                            for question in questions
                        ]
                    targets = batch["target"]

                    inputs = tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding="longest",
                        truncation=True,
                    )
                    inputs = inputs.to(device)
                    generate_kwargs = dict(
                        return_dict_in_generate=True,
                        max_length=args.max_length,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                        # repetition_penalty=1.3,
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
                        outputs.sequences[:, inputs.input_ids.shape[1]:],
                        skip_special_tokens=True,
                    )

                    for question, ans_model, target in zip(questions, generations, targets):
                        mode = (
                            "multiple_choice"
                            if task in MULTIPLE_CHOICE_TASKS
                            else "free_form"
                        )
                        ans_ = extract_ans(ans_model, mode)
                        if mode == "multiple_choice":
                            gold = target[1]
                        else:
                            gold = target
                        pred = ans_
                        is_pred_true = pred == gold
                        if is_pred_true:
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
                        fd.write(
                            "Q: %s\nA_model:\n%s\nA_target:\n%s\n\n"
                            % (question, ans_, target)
                        )

        task_acc = acc / len(eval_dataset)
        evaluation_metric = EvaluationMetrics(accuracy=task_acc)
        evaluation_result = EvaluationResults(
            samples=all_samples,
            metrics=evaluation_metric,
        )

        all_ave_acc += task_acc
        num_task += 1

        logger.info("%s acc %.4f" % (task, task_acc))
        tb_writter.add_scalar(f"{task}/accuracy", task_acc, 1)
        with results_file.open("w") as handle:
            json.dump(evaluation_result.to_dict(), handle)

    all_ave_acc = all_ave_acc / num_task
    logger.info("Average Acc: %.4f" % (all_ave_acc))
    tb_writter.add_scalar("Average Acc", all_ave_acc, 1)
