from models import LlamaForCausalLMNew
from transformers import AutoTokenizer, LlamaConfig
import torch
from datasets import load_dataset


model_id = "meta-llama/Llama-2-7b-hf"
# config = LlamaConfig.from_pretrained(
#     model_id,
#     use_auth_token=True,
#     token=None,
#     use_flash_attn=False,
# )
model_kwargs = {}
model_kwargs["torch_dtype"] = torch.float16
model_kwargs["device_map"] = "auto"
model_kwargs["token"] = None
model_kwargs["cache_dir"] = "../cache"



model = LlamaForCausalLMNew.from_pretrained(
    model_id,
    device_map = "auto",
    cache_dir="../cache",
)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    cache_dir="../cache",
)
tokenizer.pad_token = tokenizer.eos_token


eval_dataset = load_dataset("gsm8k", "main", split="test")
print(eval_dataset[0])



questions = eval_dataset[0]["question"]
answers = eval_dataset[0]["answer"]
inputs = tokenizer(
    questions,
    return_tensors="pt",
    padding="longest",
    truncation=True,
)

inputs = inputs.to("cuda")
generate_kwargs = dict(
    return_dict_in_generate=True,
    max_length=None,
    max_new_tokens=1,
    output_scores=True,
    pad_token_id=tokenizer.eos_token_id,
    use_cache=True,
)

generate_kwargs["do_sample"] = True
generate_kwargs["temperature"] = 0.8
generate_kwargs["top_k"] = 50
generate_kwargs["top_p"] = 0.95

outputs = model.generate(**inputs, **generate_kwargs)