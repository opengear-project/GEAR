lm-eval --model hf --model_args pretrained=meta-llama/Llama-2-7b-hf --tasks openbookqa --compress_args compress_method="poweriteration",rank=300,rankv=5000
lm-eval --model hf --model_args pretrained=meta-llama/Llama-2-7b-hf --tasks openbookqa --compress_args compress_method="uniformquant",quantize_bit=8
