# # python eval_gsm8k_cot.py --model meta-llama/Llama-2-7b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method outquantize_with_lrap --rank 0.05 --rankv 0.05 --quantize_bit 4 --loop 3 --left 0.01 --attention_number 40 > outquantlrap_6bl0.01_r0.1.txt

# python eval_gsm8k_cot.py --model meta-llama/Llama-2-7b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method groupquantization --attention_number 40 --quantize_bit 8 > groupquant_8b.txt
# python eval_gsm8k_cot.py --model meta-llama/Llama-2-7b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method groupquantization --attention_number 40 --quantize_bit 6 > groupquant_6b.txt
# python eval_gsm8k_cot.py --model meta-llama/Llama-2-7b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method groupquantization --attention_number 40 --quantize_bit 4 > groupquant_4b.txt


python eval_gsm8k_cot.py --model meta-llama/Llama-2-13b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method uniformquantization --attention_number 40 --quantize_bit 8 > uniquant_8b.txt
python eval_gsm8k_cot.py --model meta-llama/Llama-2-13b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method uniformquantization --attention_number 40 --quantize_bit 6 > uniquant_6b.txt
python eval_gsm8k_cot.py --model meta-llama/Llama-2-13b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method uniformquantization --attention_number 40 --quantize_bit 4 > uniquant_4b.txt

python eval_gsm8k_cot.py --model meta-llama/Llama-2-13b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method groupquantization --attention_number 40 --quantize_bit 8 > groupquant_8b.txt
python eval_gsm8k_cot.py --model meta-llama/Llama-2-13b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method groupquantization --attention_number 40 --quantize_bit 6 > groupquant_6b.txt
python eval_gsm8k_cot.py --model meta-llama/Llama-2-13b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method groupquantization --attention_number 40 --quantize_bit 4 > groupquant_4b.txt

python eval_gsm8k_cot.py --model meta-llama/Llama-2-13b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 8 --left 0.01 > outquant_8b_l0.01.txt
python eval_gsm8k_cot.py --model meta-llama/Llama-2-13b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 6 --left 0.01 > outquant_6b_l0.01.txt
python eval_gsm8k_cot.py --model meta-llama/Llama-2-13b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 4 --left 0.01 > outquant_4b_l0.01.txt


python eval_gsm8k_cot.py --model meta-llama/Llama-2-13b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 8 --left 0.02 > outquant_8b_l0.02.txt
python eval_gsm8k_cot.py --model meta-llama/Llama-2-13b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 6 --left 0.02 > outquant_6b_l0.02.txt
python eval_gsm8k_cot.py --model meta-llama/Llama-2-13b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 4 --left 0.02 > outquant_4b_l0.02.txt

python eval_gsm8k_cot.py --model meta-llama/Llama-2-13b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 8 --left 0.10 > outquant_8b_l0.10.txt
python eval_gsm8k_cot.py --model meta-llama/Llama-2-13b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 6 --left 0.10 > outquant_6b_l0.10.txt
python eval_gsm8k_cot.py --model meta-llama/Llama-2-13b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 4 --left 0.10 > outquant_4b_l0.10.txt

# python eval_gsm8k_cot.py --model meta-llama/Llama-2-13b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method outquantize_with_lrap --attention_number 40 --quantize_bit 8 --left 0.02 > outquant_8b_l0.02.txt
python eval_gsm8k_cot.py --model meta-llama/Llama-2-13b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method outquantize_with_lrap --attention_number 40 --quantize_bit 6 --left 0.02 --rank 0.1 --rankv 0.1 --loop 3 > outquantwithlrap_6b_l0.02_r0.1.txt
python eval_gsm8k_cot.py --model meta-llama/Llama-2-13b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method outquantize_with_lrap --attention_number 40 --quantize_bit 4 --left 0.02 --rank 0.05 --rankv 0.05 --loop 3  > outquantwithlrap_4b_l0.02_r0.05.txt

python eval_gsm8k_cot.py --model meta-llama/Llama-2-13b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method outquantize_with_lrap --attention_number 40 --quantize_bit 6 --left 0.01 --rank 0.1 --rankv 0.1 --loop 3 > outquantwithlrap_6b_l0.01_r0.1.txt
python eval_gsm8k_cot.py --model meta-llama/Llama-2-13b-hf --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method outquantize_with_lrap --attention_number 40 --quantize_bit 4 --left 0.01 --rank 0.05 --rankv 0.05 --loop 3 > outquantwithlrap_4b_l0.01_r0.05.txt

