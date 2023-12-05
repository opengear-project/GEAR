import transformers
import torch
import argparse
from dataset import create_dataset
from models import create_model
from tqdm import tqdm
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(
    description="A pattern for evalating model on specific datasets"
)
parser.add_argument(
    "--model",
    type=str,
    # default="Salesforce/xgen-7b-8k-inst",
    default="togethercomputer/Llama-2-7B-32K-Instruct"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="EdinburghNLP/xsum",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=4,
)
parser.add_argument(
    "--maxlength",
    type=int,
    default=4000,
)
parser.add_argument(
    "--offset",
    type=int,
    default=50,
)
parser.add_argument(
    "--max-new-tokens",
    type=int,
    default=64,
)
parser.add_argument("--cache_dir", type=str, default="./cache/")
args = parser.parse_args()

import evaluate

rouge = evaluate.load("rouge")
delimiter = "[/INST]\n\n"


def main(args):
    # create dataset
    val_dataset, test_dataset = create_dataset(args)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # create model
    model, tokenizer = create_model(args)
    # model = model.to("cuda:0")
    tokenizer = tokenizer
    rouge1 = 0.0
    rouge2 = 0.0
    rougeL = 0.0
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, desc="Processing")):
            if args.dataset == "EdinburghNLP/xsum":
                article = data["document"]
            elif args.dataset == "kmfoda/booksum":
                article = data["chapter"]
            # prompt = (
            #     f"### Human: Please summarize the following article.\n\n{article}.\n###"
            # )
            # prompt = (f"[INST]\nPlease summarize the following article.\n\n{article}.\n\n[/INST]\n\n")
            prompt = [(f"[INST]\nPlease summarize the following article.\n\n{article[doc_id][0:int(args.maxlength-args.offset)]}.\n\n[/INST]\n\n") for doc_id in range(len(article))]
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=args.maxlength)
            # TODO search parameter unsured
            if args.dataset == "EdinburghNLP/xsum":
                reference = data["summary"]
            elif args.dataset == "kmfoda/booksum":
                reference = data["summary_text"]
                # for i in range(len(reference)):
                #     print(len(reference[i]))
            sample = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                temperature=0.7, repetition_penalty=1.1, top_p=0.7, top_k=50,eos_token_id=50256,do_sample=True)
            output = tokenizer.batch_decode(sample)
            # print(reference)

            predictions = [output[doc_id].split(delimiter)[1].strip() for doc_id in range(len(article))]
            results = rouge.compute(
                predictions=predictions,
                references=reference,
                use_aggregator=True,
            )
            
            rouge1 += results["rouge1"]
            rouge2 += results["rouge2"]
            rougeL += results["rougeL"]
            if i % 10 == 0:
                print(
                    f"rouge1: {rouge1/(i+1)}, rouge2: {rouge2/(i+1)}, rougeL: {rougeL/(i+1)}"
                )
        print(
            f"rouge1_avg: {rouge1/len(test_dataset)}, rouge2_avg: {rouge2/len(test_dataset)}, rougeL_avg: {rougeL/len(test_dataset)}"
        )


if __name__ == "__main__":
    main(args)
