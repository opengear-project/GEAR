### gsm8k
python evaluation_gsm8k.py --model meta-llama/Llama-2-13b --prompt_file gsm8k_prompt_original.txt --batch_size 6 --max_new_tokens 256 --compress_method KCVT --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 20

python evaluation_gsm8k.py --model meta-llama/Llama-2-13b --prompt_file gsm8k_prompt_original.txt --batch_size 6 --max_new_tokens 256 --compress_method GEAR --attention_number 40 --quantize_bit 2 --group_size 64 --loop 3 --prefillrank 4 --prefillrankv 4 --rank 2 --rankv 2 --left 0.02 --streaming --streaming_gap 64

python evaluation_gsm8k.py --model meta-llama/Llama-2-13b --prompt_file gsm8k_prompt_original.txt --batch_size 6 --max_new_tokens 256 --compress_method GEAR-KCVT --attention_number 40 --quantize_bit 4 --loop 3 --prefillrank 4 --prefillrankv 4 --rank 2 --rankv 2 --left 0.02 --streaming --streaming_gap 64

python evaluation_gsm8k.py --model meta-llama/Llama-2-13b --prompt_file gsm8k_prompt_original.txt --batch_size 6 --max_new_tokens 256 --compress_method GEARL --attention_number 40 --quantize_bit 2 --group_size 64 --loop 3 --prefillrank 4 --prefillrankv 4 --rank 2 --rankv 2 --streaming --streaming_gap 64

python evaluation_gsm8k.py --model meta-llama/Llama-2-13b --prompt_file gsm8k_prompt_original.txt --batch_size 6 --max_new_tokens 256 --compress_method GEARL-KCVT --attention_number 40 --quantize_bit 4 --loop 3 --prefillrank 4 --prefillrankv 4 --rank 2 --rankv 2 --streaming --streaming_gap 64

###aqua
python evaluation_aqua_cot.py --model meta-llama/Llama-2-13b --batch_size 6 --max_new_tokens 196 --model_max_length 4096 --root_output_dir ./aqua --compress_method KCVT --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 20

python evaluation_aqua_cot.py --model meta-llama/Llama-2-13b --batch_size 6 --max_new_tokens 196 --model_max_length 4096 --root_output_dir ./aqua --compress_method GEAR --attention_number 40 --quantize_bit 2 --group_size 64 --loop 3 --prefillrank 4 --prefillrankv 4 --rank 2 --rankv 2 --left 0.02 --streaming --streaming_gap 64

python evaluation_aqua_cot.py --model meta-llama/Llama-2-13b --batch_size 6 --max_new_tokens 196 --model_max_length 4096 --root_output_dir ./aqua --compress_method GEAR-KCVT --attention_number 40 --quantize_bit 4 --loop 3 --prefillrank 4 --prefillrankv 4 --rank 2 --rankv 2 --left 0.02 --streaming --streaming_gap 64

python evaluation_aqua_cot.py --model meta-llama/Llama-2-13b --batch_size 6 --max_new_tokens 196 --model_max_length 4096 --root_output_dir ./aqua --compress_method GEARL --attention_number 40 --quantize_bit 2 --group_size 64 --loop 3 --prefillrank 4 --prefillrankv 4 --rank 2 --rankv 2 --streaming --streaming_gap 64

python evaluation_aqua_cot.py --model meta-llama/Llama-2-13b --batch_size 6 --max_new_tokens 196 --model_max_length 4096 --root_output_dir ./aqua --compress_method GEARL-KCVT --attention_number 40 --quantize_bit 4 --loop 3 --prefillrank 4 --prefillrankv 4 --rank 2 --rankv 2 --streaming --streaming_gap 64

### bbh
python evaluation_bbh_cot.py --model meta-llama/Llama-2-13b --batch_size 4 --max_new_tokens 256 --compress_method KCVT --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 20

python evaluation_bbh_cot.py --model meta-llama/Llama-2-13b --batch_size 4 --max_new_tokens 256 --compress_method GEAR --attention_number 40 --quantize_bit 2 --group_size 64 --loop 3 --prefillrank 4 --prefillrankv 4 --rank 2 --rankv 2 --left 0.02 --streaming --streaming_gap 64

python evaluation_bbh_cot.py --model meta-llama/Llama-2-13b --batch_size 4 --max_new_tokens 256 --compress_method GEAR-KCVT --attention_number 40 --quantize_bit 4 --loop 3 --prefillrank 4 --prefillrankv 4 --rank 2 --rankv 2 --left 0.02 --streaming --streaming_gap 64

python evaluation_bbh_cot.py --model meta-llama/Llama-2-13b --batch_size 4 --max_new_tokens 256 --compress_method GEARL --attention_number 40 --quantize_bit 2 --group_size 64 --loop 3 --prefillrank 4 --prefillrankv 4 --rank 2 --rankv 2 --streaming --streaming_gap 64











