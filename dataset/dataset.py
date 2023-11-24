from datasets import load_dataset

def create_dataset(args):
    val_dataset = load_dataset(
        "ccdv/arxiv-summarization", split="validation", cache_dir="./cache/"
    )
    test_dataset = load_dataset(
        "ccdv/arxiv-summarization", split="test", cache_dir="./cache/"
    )
    return val_dataset, test_dataset