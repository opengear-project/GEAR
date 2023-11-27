import transformers
import torch
import argparse
from dataset import create_dataset
from models import create_model
parser = argparse.ArgumentParser(description="A pattern for evalating model on specific datasets")
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
    default=1,
)
parser.add_argument("--cache_dir", type=str, default="./cache/")
args = parser.parse_args()
def main(args):
    #create dataset
    val_dataset, test_dataset = create_dataset(
        args
    )
    #create model
    model, tokenizer = create_model(args)
    # model = model.to("cuda:0")
    tokenizer = tokenizer
    with torch.no_grad():
        for i, data in enumerate(test_dataset):
            article = data["article"]
            abstract = data["abstract"]
            prompt = f"### Human: Please summarize the following article.\n\n{article}.\n###"
            inputs = tokenizer(prompt, return_tensors="pt")
            print(inputs)
            inputs = inputs.to("cuda:0")
            sample = model.generate(**inputs, do_sample=True, max_new_tokens=2048, top_k=100, eos_token_id=50256)
            output = tokenizer.decode(sample[0])
            print(output)
            break
        
        


















if __name__ == "__main__":
    main(args)