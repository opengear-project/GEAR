# python3 test_gsm8k.py --model_name_or_path LoftQ/Llama-2-7b-hf-fp16-64rank-gsm8k --batch_size 8 --device_map "cuda:0" --gpu 0 --compress_method dynamicpoweriteration_intermediate --rank 0.5 --rankv 0 --attention_number 32 --loop 3 --start_saving 0.05 --locality_saving 0.05 --stage 100 > llama2_gsm8k_krank0.5_inter_0.05+0.05.txt
# python3 test_gsm8k.py --model_name_or_path LoftQ/Llama-2-7b-hf-fp16-64rank-gsm8k --batch_size 8 --device_map "cuda:0" --gpu 0 --compress_method dynamicpoweriteration_intermediate --rank 0.5 --rankv 0 --attention_number 32 --loop 3 --start_saving 0.10 --locality_saving 0.10 --stage 100 > llama2_gsm8k_krank0.5_inter_0.10+0.10.txt
# python3 test_gsm8k.py --model_name_or_path LoftQ/Llama-2-7b-hf-fp16-64rank-gsm8k --batch_size 8 --device_map "cuda:0" --gpu 0 --compress_method densesparseuniformquantization --attention_number 32 --quantize_bit 6 --left 0.05 > llama2_gsm8k_outquant6_0.05.txt
# python3 test_gsm8k.py --model_name_or_path LoftQ/Llama-2-7b-hf-fp16-64rank-gsm8k --batch_size 8 --device_map "cuda:0" --gpu 0 --compress_method uniformquantization --attention_number 32 --quantize_bit 6 --stage 100 > llama2_gsm8k_quant6.txt
# python3 test_gsm8k.py --model_name_or_path LoftQ/Llama-2-7b-hf-fp16-64rank-gsm8k --batch_size 8 --device_map "cuda:0" --gpu 0 --compress_method groupquantization --attention_number 32 --quantize_bit 6 --stage 100 > llama2_gsm8k_groupquant6.txt




# python3 test_gsm8k.py --model_name_or_path LoftQ/Llama-2-7b-hf-fp16-64rank-gsm8k --batch_size 8 --device_map "cuda:0" --gpu 0 --compress_method quantize_with_lrap --attention_number 32 --quantize_bit 6 --rank 20 --rankv 40 --loop 3

# python3 test_gsm8k.py --model_name_or_path LoftQ/Llama-2-7b-hf-fp16-64rank-gsm8k --batch_size 8 --device_map "cuda:0" --gpu 0 --compress_method densesparseuniformquantization --attention_number 32 --quantize_bit 4 --left 0.02 > llama2_gsm8k_outquant4_0.02.txt

# python3 test_gsm8k.py --model_name_or_path LoftQ/Llama-2-7b-hf-fp16-64rank-gsm8k --batch_size 8 --device_map "cuda:0" --gpu 0 --compress_method quantize_with_lrap --attention_number 32 --quantize_bit 6 --rank 0.1 --rankv 0.1 --loop 3 > llama2_gsm8k_uniquant6_0.02.txt
python3 test_gsm8k.py --model_name_or_path LoftQ/Llama-2-7b-hf-fp16-64rank-gsm8k --batch_size 8 --device_map "cuda:0" --gpu 0 --compress_method outquantize_with_lrap --attention_number 32 --quantize_bit 4 --rank 0.05 --rankv 0.05 --loop 3 --left 0.02 > llama2_gsm8k_outquant4_0.02_lrap_0.05.txt
# python3 test_gsm8k.py --model_name_or_path LoftQ/Llama-2-7b-hf-fp16-64rank-gsm8k --batch_size 8 --device_map "cuda:0" --gpu 0 --compress_method outquantize_with_lrap --attention_number 32 --quantize_bit 4 --rank 0.05 --rankv 0.10 --loop 3 --left 0.02 > llama2_gsm8k_outquant4_0.02_lrap_0.05.txt


