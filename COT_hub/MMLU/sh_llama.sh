# python run_mmlu_open_source.py --model meta-llama/Llama-2-7b-hf --ntrain 0 --batch_size 12 --attention_number 40 > zeroshot_llama_benchmark.txt

# python run_mmlu_open_source.py --model meta-llama/Llama-2-7b-hf --ntrain 0 --batch_size 12 --compress_method uniformquantization --attention_number 40 --quantize_bit 6 --streaming --streaming_gap 40 > zeroshot_llama_unifomrQ6b.txt
# python run_mmlu_open_source.py --model meta-llama/Llama-2-7b-hf --ntrain 0 --batch_size 12 --compress_method uniformquantization --attention_number 40 --quantize_bit 8 --streaming --streaming_gap 40 > zeroshot_llama_unifomrQ8b.txt
# python run_mmlu_open_source.py --model meta-llama/Llama-2-7b-hf --ntrain 0 --batch_size 12 --compress_method uniformquantization --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 40 > zeroshot_llama_unifomrQ4b.txt


# python run_mmlu_open_source.py --model meta-llama/Llama-2-7b-hf --ntrain 0 --batch_size 12 --compress_method groupquantization --attention_number 40 --quantize_bit 6 --streaming --streaming_gap 40 > zeroshot_llama_groupQ6b.txt
# python run_mmlu_open_source.py --model meta-llama/Llama-2-7b-hf --ntrain 0 --batch_size 12 --compress_method groupquantization --attention_number 40 --quantize_bit 8 --streaming --streaming_gap 40 > zeroshot_llama_groupQ8b.txt
# python run_mmlu_open_source.py --model meta-llama/Llama-2-7b-hf --ntrain 0 --batch_size 12 --compress_method groupquantization --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 40 > zeroshot_llama_groupQ4b.txt

# python run_mmlu_open_source.py --model meta-llama/Llama-2-7b-hf --ntrain 0 --batch_size 12 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 6 --streaming --streaming_gap 40 --left 0.02 > zeroshot_llama_outlierQ6b_0.02.txt
# python run_mmlu_open_source.py --model meta-llama/Llama-2-7b-hf --ntrain 0 --batch_size 12 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 8 --streaming --streaming_gap 40 --left 0.02 > zeroshot_llama_outlierQ8b_0.02.txt
# python run_mmlu_open_source.py --model meta-llama/Llama-2-7b-hf --ntrain 0 --batch_size 12 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 40 --left 0.02 > zeroshot_llama_outlierQ4b_0.02.txt


# python run_mmlu_open_source.py --model meta-llama/Llama-2-7b-hf --ntrain 0 --batch_size 12 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 6 --streaming --streaming_gap 40 --left 0.05 > zeroshot_llama_outlierQ6b_0.05.txt
# python run_mmlu_open_source.py --model meta-llama/Llama-2-7b-hf --ntrain 0 --batch_size 12 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 8 --streaming --streaming_gap 40 --left 0.05 > zeroshot_llama_outlierQ8b_0.05.txt
# python run_mmlu_open_source.py --model meta-llama/Llama-2-7b-hf --ntrain 0 --batch_size 12 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 40 --left 0.05 > zeroshot_llama_outlierQ4b_0.05.txt

# python run_mmlu_open_source.py --model meta-llama/Llama-2-7b-hf --ntrain 0 --batch_size 12 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 6 --streaming --streaming_gap 40 --left 0.10 > zeroshot_llama_outlierQ6b_0.10.txt
# python run_mmlu_open_source.py --model meta-llama/Llama-2-7b-hf --ntrain 0 --batch_size 12 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 8 --streaming --streaming_gap 40 --left 0.10 > zeroshot_llama_outlierQ8b_0.10.txt
# python run_mmlu_open_source.py --model meta-llama/Llama-2-7b-hf --ntrain 0 --batch_size 12 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 40 --left 0.10 > zeroshot_llama_outlierQ4b_0.10.txt


# python run_mmlu_open_source.py --model meta-llama/Llama-2-7b-hf --ntrain 0 --batch_size 12 --compress_method outquantize_with_lrap --attention_number 40 --quantize_bit 6 --streaming --streaming_gap 40 --left 0.05  --rank 0.05 --rankv 0.05 --loop 3 > zeroshot_llama_outlierQ6b_0.05_lrap_0.05.txt
# python run_mmlu_open_source.py --model meta-llama/Llama-2-7b-hf --ntrain 0 --batch_size 12 --compress_method outquantize_with_lrap --attention_number 40 --quantize_bit 8 --streaming --streaming_gap 40 --left 0.05 --rank 0.05 --rankv 0.05 --loop 3 > zeroshot_llama_outlierQ8b_0.05_lrap_0.05.txt
# python run_mmlu_open_source.py --model meta-llama/Llama-2-7b-hf --ntrain 0 --batch_size 12 --compress_method outquantize_with_lrap --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 40 --left 0.05  --rank 0.05 --rankv 0.05 --loop 3 > zeroshot_llama_outlierQ4b_0.05_lrap_0.05.txt


python run_mmlu_open_source.py --model meta-llama/Llama-2-7b-hf --ntrain 0 --batch_size 12 --compress_method fake_quant_with_lrap --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 40 --rank 0.01 --rankv 0.01 --loop 3 > test1.txt

python run_mmlu_open_source.py --model meta-llama/Llama-2-7b-hf --ntrain 0 --batch_size 12 --compress_method fake_quant_with_lrap --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 40 --rank 0.01 --rankv 0.01 --loop 3 > test2.txt