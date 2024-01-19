python eval_gsm8k_cot.py --model Qwen/Qwen-7B --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --attention_number 40 > qwen7b_cot.txt

python eval_gsm8k_cot.py --model Qwen/Qwen-7B --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method uniformquantization --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 40 > qwen7b_cot_uniQ4.txt
python eval_gsm8k_cot.py --model Qwen/Qwen-7B --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method uniformquantization --attention_number 40 --quantize_bit 6 --streaming --streaming_gap 40 > qwen7b_cot_uniQ6.txt
python eval_gsm8k_cot.py --model Qwen/Qwen-7B --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method uniformquantization --attention_number 40 --quantize_bit 8 --streaming --streaming_gap 40 > qwen7b_cot_uniQ8.txt


python eval_gsm8k_cot.py --model qwen/Qwen-7B --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method groupquantization --attention_number 40 --quantize_bit 8 --streaming --streaming_gap 40 > qwen7b_groupquant_8b.txt
python eval_gsm8k_cot.py --model qwen/Qwen-7B --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method groupquantization --attention_number 40 --quantize_bit 6 --streaming --streaming_gap 40 > qwen7b_groupquant_6b.txt
python eval_gsm8k_cot.py --model qwen/Qwen-7B --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method groupquantization --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 40 > qwen7b_groupquant_4b.txt


python eval_gsm8k_cot.py --model qwen/Qwen-7B --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 8 --left 0.05 --streaming --streaming_gap 40 > qwen7b_outquant_8b_l0.05.txt
python eval_gsm8k_cot.py --model qwen/Qwen-7B --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 6 --left 0.05 --streaming --streaming_gap 40 > qwen7b_outquant_6b_l0.05.txt
python eval_gsm8k_cot.py --model qwen/Qwen-7B --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 4 --left 0.05 --streaming --streaming_gap 40 > qwen7b_outquant_4b_l0.05.txt


python eval_gsm8k_cot.py --model qwen/Qwen-7B --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 8 --left 0.02 --streaming --streaming_gap 40 > qwen7b_outquant_8b_l0.02.txt
python eval_gsm8k_cot.py --model qwen/Qwen-7B --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 6 --left 0.02 --streaming --streaming_gap 40 > qwen7b_outquant_6b_l0.02.txt
python eval_gsm8k_cot.py --model qwen/Qwen-7B --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 4 --left 0.02 --streaming --streaming_gap 40 > qwen7b_outquant_4b_l0.02.txt



python eval_gsm8k_cot.py --model qwen/Qwen-7B --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method outquantize_with_lrap --attention_number 40 --quantize_bit 4 --left 0.02 --rank 0.1 --rankv 0.1 --loop 3 > qwen7b_outquantwithlrap_4b_l0.02_r0.1.txt
python eval_gsm8k_cot.py --model qwen/Qwen-7B --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method outquantize_with_lrap --attention_number 40 --quantize_bit 4 --left 0.01 --rank 0.1 --rankv 0.1 --loop 3 > qwen7b_outquantwithlrap_4b_l0.01_r0.1.txt
python eval_gsm8k_cot.py --model qwen/Qwen-7B --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method outquantize_with_lrap --attention_number 40 --quantize_bit 4 --left 0.02 --rank 0.05 --rankv 0.05 --loop 3  > qwen7b_outquantwithlrap_4b_l0.02_r0.05.txt


python eval_gsm8k_cot.py --model qwen/Qwen-7B --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method outquantize_with_lrap --attention_number 40 --quantize_bit 6 --left 0.02 --rank 0.1 --rankv 0.1 --loop 3 > qwen7b_outquantwithlrap_6b_l0.02_r0.1.txt
python eval_gsm8k_cot.py --model qwen/Qwen-7B --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method outquantize_with_lrap --attention_number 40 --quantize_bit 6 --left 0.01 --rank 0.1 --rankv 0.1 --loop 3 > qwen7b_outquantwithlrap_6b_l0.01_r0.1.txt
python eval_gsm8k_cot.py --model qwen/Qwen-7B --prompt_file prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method outquantize_with_lrap --attention_number 40 --quantize_bit 6 --left 0.02 --rank 0.05 --rankv 0.05 --loop 3  > qwen7b_outquantwithlrap_6b_l0.02_r0.05.txt