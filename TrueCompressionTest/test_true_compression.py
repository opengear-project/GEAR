from models import TrueLlamaForCausalLMNew
from transformers import AutoTokenizer
import torch
compress_config = {}
compress_config["compress_mode"] = "gear"
compress_config["quantize_bit"] = 4
compress_config["left"] = 0.02
compress_config["rank"] = 20
compress_config["loop"] = 0
model = TrueLlamaForCausalLMNew.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    cache_dir="../cache",
    device_map = "auto",
    compress_config = compress_config
)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    token=None,
    padding_side="left",
    model_max_length=2000,
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
inputs["input_ids"] = torch.randint(30000,(16,1000)).long().to("cuda")
import time
start = time.time()
outputs = model.generate(input_ids = inputs["input_ids"], max_length=1200,use_cache=True)
torch.cuda.synchronize()
end = time.time()
results = tokenizer.decode(outputs[0],skip_special_tokens=True)
print(end - start)