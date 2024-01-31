from models import TrueLlamaForCausalLMNew,LlamaForCausalLM,LlamaForCausalLMNew
from transformers import AutoTokenizer
import torch
seed = 2345
#### we use llamaattention instead of SDPAttention to allow 16bits inference
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
compress_config = {}
compress_config["compress_mode"] = "outlier_batch"
compress_config["quantize_bit"] = 4
compress_config["left"] = 0.10
compress_config["rank"] = 20
compress_config["loop"] = 0
compress_config["stream"] = True
compress_config["streaming_gap"] = 20
model = TrueLlamaForCausalLMNew.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    cache_dir="../cache",
    device_map = "auto",
    compress_config = compress_config,
    torch_dtype = torch.float16,
    # torch_dtype = torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    token=None,
    padding_side="left",
    model_max_length=2000,
    use_fast=False,
    cache_dir="../cache",
    max_length=2000,
)
tokenizer.pad_token = tokenizer.eos_token
sentence = "what is this thing mainly about?"
sentence_group = []
for i in range(30):
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
outputs = model.generate(**inputs, max_length=2100,use_cache=True)
torch.cuda.synchronize()
end = time.time()
results = tokenizer.decode(outputs[0],skip_special_tokens=True)
print(end - start)
peak_memory = torch.cuda.max_memory_allocated(device="cuda") / (1024**2)  # 转换为MB单位

print(f"Peak memory usage on GPU: {peak_memory} MB")