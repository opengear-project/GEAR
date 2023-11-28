import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig


def create_model(args):
    # 8bit quant
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir="../cache/",
        device_map="auto",
        padding=True,
        truncation=True,
        use_cache=True,
        max_length=8000,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
        use_cache=True,
        device_map="auto",
    )
    # model = model.half()
    return model, tokenizer
