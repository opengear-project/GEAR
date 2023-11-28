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
    default="Salesforce/xgen-7b-8k-inst",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="EdinburghNLP/xsum",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=6,
)
parser.add_argument(
    "--maxlength",
    type=int,
    default=8000,
)
parser.add_argument("--cache_dir", type=str, default="./cache/")
args = parser.parse_args()

import evaluate

rouge = evaluate.load("rouge")
delimiter = "### Assistant:"


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
            article = data["document"]
            prompt = (
                f"### Human: Please summarize the following article.\n\n{article}.\n###"
            )
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = inputs.to("cuda:0")
            sample = model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=500,
                top_k=100,
                eos_token_id=50256,
            )
            output = tokenizer.decode(sample[0])
            results = rouge.compute(
                predictions=[output.split(delimiter)[1].strip()],
                references=[data["summary"]],
                use_aggregator=True,
            )
            rouge1 += results["rouge1"]
            rouge2 += results["rouge2"]
            rougeL += results["rougeL"]
            if i % 100 == 0:
                print(
                    f"rouge1: {rouge1/(i+1)}, rouge2: {rouge2/(i+1)}, rougeL: {rougeL/(i+1)}"
                )
        print(
            f"rouge1: {rouge1/len(test_dataset)}, rouge2: {rouge2/len(test_dataset)}, rougeL: {rougeL/len(test_dataset)}"
        )


if __name__ == "__main__":
    main(args)
