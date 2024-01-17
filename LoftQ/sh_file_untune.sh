

python3 test_gsm8k.py --model_name_or_path meta-llama/Llama-2-7b-hf --batch_size 8 --device_map "cuda:0" --gpu 0 > benchmark_untune.txt

python3 test_gsm8k.py --model_name_or_path meta-llama/Llama-2-7b-hf --batch_size 8 --device_map "cuda:0" --gpu 0 --compress_method densesparseuniformquantization --attention_number 32 --quantize_bit 6 --left 0.02 > llama2_gsm8k_outquant6_0.02.txt
python3 test_gsm8k.py --model_name_or_path meta-llama/Llama-2-7b-hf --batch_size 8 --device_map "cuda:0" --gpu 0 --compress_method uniformquantization --attention_number 32 --quantize_bit 6 --stage 100 > llama2_gsm8k_quant6.txt
python3 test_gsm8k.py --model_name_or_path meta-llama/Llama-2-7b-hf --batch_size 8 --device_map "cuda:0" --gpu 0 --compress_method uniformquantization --attention_number 32 --quantize_bit 4 --stage 100 > llama2_gsm8k_quant4.txt
python3 test_gsm8k.py --model_name_or_path meta-llama/Llama-2-7b-hf --batch_size 8 --device_map "cuda:0" --gpu 0 --compress_method densesparseuniformquantization --attention_number 32 --quantize_bit 4 --left 0.02 > llama2_gsm8k_outquant4_0.02.txt
python3 test_gsm8k.py --model_name_or_path meta-llama/Llama-2-7b-hf --batch_size 8 --device_map "cuda:0" --gpu 0 --compress_method groupquantization --attention_number 32 --quantize_bit 6 --stage 100 > llama2_gsm8k_groupquant6.txt




# python3 test_gsm8k.py --model_name_or_path meta-llama/Llama-2-7b-hf --batch_size 8 --device_map "cuda:0" --gpu 0 --compress_method quantize_with_lrap --attention_number 32 --quantize_bit 6 --rank 20 --rankv 40 --loop 3



# python3 test_gsm8k.py --model_name_or_path meta-llama/Llama-2-7b-hf --batch_size 8 --device_map "cuda:0" --gpu 0 --compress_method quantize_with_lrap --attention_number 32 --quantize_bit 6 --rank 0.1 --rankv 0.1 --loop 3 > llama2_gsm8k_uniquant6_0.02.txt


# python3 test_gsm8k.py --model_name_or_path meta-llama/Llama-2-7b-hf --batch_size 8 --device_map "cuda:0" --gpu 0 --compress_method densesparseuniformquantization --attention_number 32 --quantize_bit 4 --left 0.01 > llama2_gsm8k_outquant4_0.01.txt

