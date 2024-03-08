from h2o_llama_self_written import LlamaForCausalLMH2O
from compress_config import GPT2CompressConfig
from transformers import AutoTokenizer


model_id = "meta-llama/Llama-2-7b-hf"
config = GPT2CompressConfig(
    compress_method="h2o",
)

model = LlamaForCausalLMH2O.from_pretrained(
    model_id,
    # config=config,
    cache_dir="../cache",
    compress_config=config,
    device_map="cpu",
)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=None,
    padding_side="left",
    model_max_length=256,
    use_fast=False,
    cache_dir="../cache",
)
tokenizer.pad_token = tokenizer.eos_token

Question = ["what is this thing mainly about?"]
inputs = tokenizer(
    Question,
    return_tensors="pt",
    padding="longest",
    truncation=True,
)
inputs["input_ids"] = inputs["input_ids"]
generated = model.generate(**inputs, max_length=256)
results = tokenizer.batch_decode(generated, skip_special_tokens=True)
print(results)
