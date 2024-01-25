from models import TrueLlamaForCausalLMNew
from transformers import AutoTokenizer
import torch
compress_config = {}
compress_config["compress_mode"] = "uniform"
compress_config["quantize_bit"] = 4
compress_config["min"] = 0.0
compress_config["step"] = 0.0
compress_config["left"] = 0.0
compress_config["rank"] = 0.0
compress_config["loop"] = 0.0
model = TrueLlamaForCausalLMNew.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    cache_dir="../cache",
    device_map = "auto",
    compress_config = None
)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    token=None,
    padding_side="left",
    model_max_length=256,
    use_fast=False,
    cache_dir="../cache",
)
tokenizer.pad_token = tokenizer.eos_token
sentence = ["what is this thing mainly about?"]
inputs = tokenizer(
    sentence,
    return_tensors="pt",
    padding="longest",
    truncation=True,
)
inputs = inputs.to("cuda")
print(inputs["input_ids"])
inputs["input_ids"] = torch.randint(30000,(4,100)).long().to("cuda")
outputs = model.generate(input_ids = inputs["input_ids"], max_length=256)

results = tokenizer.decode(outputs[0],skip_special_tokens=True)
print(results)