# # python eval_gsm8k_cot.py --model meta-llama/Llama-2-7b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method outquantize_with_lrap --rank 0.05 --rankv 0.05 --quantize_bit 4 --loop 3 --left 0.01 --attention_number 32 > outquantlrap_6bl0.01_r0.1.txt

# python eval_gsm8k_cot.py --model meta-llama/Llama-2-7b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method groupquantization --attention_number 32 --quantize_bit 8 > groupquant_8b.txt
# python eval_gsm8k_cot.py --model meta-llama/Llama-2-7b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method groupquantization --attention_number 32 --quantize_bit 6 > groupquant_6b.txt
# python eval_gsm8k_cot.py --model meta-llama/Llama-2-7b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method groupquantization --attention_number 32 --quantize_bit 4 > groupquant_4b.txt


python eval_gsm8k_cot.py --model meta-llama/Llama-2-7b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method uniformquantization --attention_number 32 --quantize_bit 8 > uniquant_8b.txt
python eval_gsm8k_cot.py --model meta-llama/Llama-2-7b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method uniformquantization --attention_number 32 --quantize_bit 6 > uniquant_6b.txt
python eval_gsm8k_cot.py --model meta-llama/Llama-2-7b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method uniformquantization --attention_number 32 --quantize_bit 4 > uniquant_4b.txt