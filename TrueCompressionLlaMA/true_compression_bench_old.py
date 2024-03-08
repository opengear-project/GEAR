# from old_models.modeling_llama_old import LlamaForCausalLM
from GEARLM import GearLlamaForCausalLM
from transformers import AutoTokenizer
import torch
from datasets import load_dataset

seed = 3125
#### we use llamaattention instead of SDPAttention to allow 16bits inference
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
compress_config = {}
compress_config["compress_mode"] = "gear_tokenwiseQ_nopq"
compress_config["quantize_bit"] = 4
compress_config["left"] = 0.02
compress_config["rank"] = 0.02  # 0.01
compress_config["loop"] = 3
compress_config["stream"] = True
compress_config["streaming_gap"] = 20
batch_size = 20
max_length = 2000
max_token = 2000
max_generation_length = 2200
device_map = "auto"
model = GearLlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    cache_dir="../cache",
    device_map=device_map,
    compress_config=compress_config,
    torch_dtype=torch.float16,
    # torch_dtype = torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    token=None,
    padding_side="left",
    model_max_length=max_length,
    use_fast=False,
    cache_dir="../cache",
    max_length=max_length,
)
tokenizer.pad_token = tokenizer.eos_token
test = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
text_combined = test["text"]
# print(len(text_combined[0]))
sentence_group = []
for i in range(batch_size):
    # sentence_group.append(str(text_combined[i*max_token:(i+1)*max_token]))
    sentence_group.append(str(text_combined[0:max_token]))

inputs = tokenizer(
    sentence_group,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
)
inputs = inputs.to("cuda:0")
print(inputs["input_ids"].shape)
import time

start = time.time()
outputs = model.generate(**inputs, max_length=max_generation_length, use_cache=True)
torch.cuda.synchronize()
end = time.time()
results = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(end - start)
peak_memory = torch.cuda.max_memory_allocated(device="cuda") / (1024**2)  # 转换为MB单位

print(f"Peak memory usage on GPU: {peak_memory} MB")
