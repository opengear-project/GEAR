from models import TrueLlamaForCausalLMNew,LlamaForCausalLM,LlamaForCausalLMNew
from transformers import AutoTokenizer
import torch
seed = 2345
#### we use llamaattention instead of SDPAttention to allow 16bits inference
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
compress_config = {}
compress_config["compress_mode"] = "uniform_batch"
compress_config["quantize_bit"] = 4
compress_config["left"] = 0.02
compress_config["rank"] = 20
compress_config["loop"] = 0
model = TrueLlamaForCausalLMNew.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    cache_dir="../cache",
    device_map = "auto",
    compress_config = compress_config,
    # torch_dtype = torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    token=None,
    padding_side="left",
    model_max_length=1000,
    use_fast=False,
    cache_dir="../cache",
    max_length=1000,
)
tokenizer.pad_token = tokenizer.eos_token
sentence = "what is this thing mainly about?"
sentence_group = []
for i in range(108):
    sentence_group.append(sentence)
inputs = tokenizer(
    sentence_group,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
)
inputs = inputs.to("cuda")
print(inputs["input_ids"].shape)
import time
start = time.time()
# from torch.cuda.amp import autocast
# with autocast():
outputs = model.generate(**inputs, max_length=1100,use_cache=True)
torch.cuda.synchronize()
end = time.time()
results = tokenizer.decode(outputs[0],skip_special_tokens=True)
print(end - start)
peak_memory = torch.cuda.max_memory_allocated(device="cuda") / (1024**2)  # 转换为MB单位

print(f"Peak memory usage on GPU: {peak_memory} MB")