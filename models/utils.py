import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

def create_model(args):
    # 8bit quant
    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_8bit_compute_dtype=torch.bfloat16
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, cache_dir = args.cache_dir, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, cache_dir = args.cache_dir, use_cache=True, device_map="auto",quantization_config=quantization_config)
    # model = model.half()
    return model, tokenizer