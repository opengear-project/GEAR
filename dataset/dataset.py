from datasets import load_dataset


def create_dataset(args):
    # some space for the new tokens and INST
    offset = 100
    if args.dataset == "EdinburghNLP/xsum":

        val_dataset = load_dataset(args.dataset, split="validation", cache_dir="./cache/")
        test_dataset = load_dataset(args.dataset, split="test", cache_dir="./cache/")

        val_dataset = val_dataset.filter(lambda x: len(x["document"]) < args.maxlength - offset)
        test_dataset = test_dataset.filter(lambda x: len(x["document"]) < args.maxlength - offset)
    elif args.dataset == "kmfoda/booksum":
        val_dataset = load_dataset(args.dataset, split="validation", cache_dir="./cache/")
        test_dataset = load_dataset(args.dataset, split="test", cache_dir="./cache/")
        # print(len(test_dataset))
        # val_dataset = val_dataset.filter(lambda x: x["chapter_length"] < args.maxlength - offset)
        # test_dataset = test_dataset.filter(lambda x: x["chapter_length"] < args.maxlength - offset)
        # # only contain chapter and summary_text
        # val_dataset = None
        new_test_dataset = []
        # print(len(test_dataset))
        for i in range(len(test_dataset)):
            new_test_dataset.append({"chapter": test_dataset[i]["chapter"], "summary_text": test_dataset[i]["summary_text"]})
        test_dataset = new_test_dataset
        # print(test_dataset[0]["chapter"])
    return val_dataset, test_dataset
