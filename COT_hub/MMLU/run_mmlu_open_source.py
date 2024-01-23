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
    if "Llama-2" or "Qwen" in args.model:
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
        model_kwargs["token"] = args.hf_token
    if "Qwen" in args.model:
        config = transformers.AutoConfig.from_pretrained(
            args.model,
            use_auth_token=True,
            token=args.hf_token,
            use_flash_attn = False,
            trust_remote_code=True,
        )
    else:
        config = transformers.AutoConfig.from_pretrained(
            args.model,
            use_auth_token=True,
            token=args.hf_token,
            trust_remote_code=True,
        )
    from models import LlamaForCausalLMNew
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
            streaming=args.streaming,
            streaming_gap=args.streaming_gap,
        )
    )
    if compress_config is not None:
        compress_config.copy_for_all_attention()
        compress_config.calculate_compress_ratio_list(4095, 4096)
    if "Llama" in args.model:
        model = LlamaForCausalLMNew.from_pretrained(
            args.model,
            config=config,
            **model_kwargs,
            cache_dir="../cache",
            compress_config=compress_config,
        )
    elif "Qwen" in args.model:
        from models import QWenLMHeadModel
        model = QWenLMHeadModel.from_pretrained(
            args.model,
            config=config,
            **model_kwargs,
            cache_dir="../cache",
            compress_config=compress_config,
            trust_remote_code=True,
        )
    elif "Mistral" in args.model:
        from models import MistralForCausalLM
        model = MistralForCausalLM.from_pretrained(
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
    if "Qwen" or "Mistral" in args.model:
        tokenizer.pad_token = tokenizer.eos_token
    # model = model.to('cuda')
    return model, tokenizer


def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts


def batch_infer(model, tokenizer, prompts, args):
    batch_size = args.batch_size
    answers = []
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        outputs = model.generate(
            **encode_inputs, max_new_tokens=1, pad_token_id=tokenizer.pad_token_id
        )
        answers.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    answers = [answer[-1] for answer in answers]
    return answers


def main(args):
    run_results = {}
    output_filename = "run_results_%s_%sb.json" % (args.model,args.compress_method)

    model, tokenizer = load(args)
    start_time = time.time()
    for task in TASKS:
        print("Testing %s ..." % task)
        records = []
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", task + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", task + "_test.csv"), header=None
        )
        for i in range(test_df.shape[0]):
            # get prompt and make sure it fits
            k = args.ntrain
            prompt_end = format_example(test_df, i, include_answer=False)
            train_prompt = gen_prompt(dev_df, task, k)
            prompt = train_prompt + prompt_end
            while len(tokenizer.tokenize(prompt)) + 1 > 2048:  # bos token
                prompt_split = prompt.split("\n\n")
                prompt_split.pop(1)
                prompt = "\n\n".join(prompt_split)
            label = test_df.iloc[i, test_df.shape[1] - 1]
            records.append({"prompt": prompt, "answer": label})

        pred_answers = batch_infer(
            model, tokenizer, [record["prompt"] for record in records], args
        )
        gold_answers = [record["answer"] for record in records]
        run_results[task] = {"pred_answers": pred_answers, "gold_answers": gold_answers}
    with open(output_filename, "w") as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)

    compute_metric(output_filename)
    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MMLU Tasks")
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-2-7b", help="Model name or path."
    )
    parser.add_argument(
        "--tasks", nargs="+", type=str, default=TASKS, help="The evaluation tasks."
    )
    parser.add_argument(
        "--prompt_file", type=str, default="lib_prompt/mmlu-cot.json", help=""
    )
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
    parser.add_argument(
        "--root_output_dir", type=str, default="outputs", help="Root output dir"
    )
    parser.add_argument("--debug", action="store_true", default=False, help="")
    parser.add_argument("--compress_method", type=str, default="None", help="")
    parser.add_argument("--rank", type=float, default=0.0, help="")
    parser.add_argument("--rankv", type=float, default=0.0, help="")
    parser.add_argument("--loop", type=int, default=0.0, help="")
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
    parser.add_argument(
        "--streaming", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--streaming_gap", type=int, default=0, help=""
    )
    parser.add_argument(
        "--ntrain", type=int, default=0, help=""
    )
    parser.add_argument('--data_dir', type=str, default='data/')
    args = parser.parse_args()

    main(args)
