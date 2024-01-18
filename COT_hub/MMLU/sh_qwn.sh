python run_mmlu_open_source.py --model qwen/Qwen-7B --ntrain 0 --batch_size 12 --attention_number 40 > zeroshot_qwen_benchmark.txt

python run_mmlu_open_source.py --model qwen/Qwen-7B --ntrain 0 --batch_size 12 --compress_method uniformquantization --attention_number 40 --quantize_bit 6 --streaming --streaming_gap 40 > zeroshot_qwen_unifomrQ6b.txt
python run_mmlu_open_source.py --model qwen/Qwen-7B --ntrain 0 --batch_size 12 --compress_method uniformquantization --attention_number 40 --quantize_bit 8 --streaming --streaming_gap 40 > zeroshot_qwen_unifomrQ8b.txt
python run_mmlu_open_source.py --model qwen/Qwen-7B --ntrain 0 --batch_size 12 --compress_method uniformquantization --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 40 > zeroshot_qwen_unifomrQ4b.txt


python run_mmlu_open_source.py --model qwen/Qwen-7B --ntrain 0 --batch_size 12 --compress_method groupquantization --attention_number 40 --quantize_bit 6 --streaming --streaming_gap 40 > zeroshot_qwen_groupQ6b.txt
python run_mmlu_open_source.py --model qwen/Qwen-7B --ntrain 0 --batch_size 12 --compress_method groupquantization --attention_number 40 --quantize_bit 8 --streaming --streaming_gap 40 > zeroshot_qwen_groupQ8b.txt
python run_mmlu_open_source.py --model qwen/Qwen-7B --ntrain 0 --batch_size 12 --compress_method groupquantization --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 40 > zeroshot_qwen_groupQ4b.txt

python run_mmlu_open_source.py --model qwen/Qwen-7B --ntrain 0 --batch_size 12 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 6 --streaming --streaming_gap 40 --left 0.02 > zeroshot_qwen_outlierQ6b_0.02.txt
python run_mmlu_open_source.py --model qwen/Qwen-7B --ntrain 0 --batch_size 12 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 8 --streaming --streaming_gap 40 --left 0.02 > zeroshot_qwen_outlierQ8b_0.02.txt
python run_mmlu_open_source.py --model qwen/Qwen-7B --ntrain 0 --batch_size 12 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 40 --left 0.02 > zeroshot_qwen_outlierQ4b_0.02.txt


python run_mmlu_open_source.py --model qwen/Qwen-7B --ntrain 0 --batch_size 12 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 6 --streaming --streaming_gap 40 --left 0.05 > zeroshot_qwen_outlierQ6b_0.05.txt
python run_mmlu_open_source.py --model qwen/Qwen-7B --ntrain 0 --batch_size 12 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 8 --streaming --streaming_gap 40 --left 0.05 > zeroshot_qwen_outlierQ8b_0.05.txt
python run_mmlu_open_source.py --model qwen/Qwen-7B --ntrain 0 --batch_size 12 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 40 --left 0.05 > zeroshot_qwen_outlierQ4b_0.05.txt

python run_mmlu_open_source.py --model qwen/Qwen-7B --ntrain 0 --batch_size 12 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 6 --streaming --streaming_gap 40 --left 0.10 > zeroshot_qwen_outlierQ6b_0.10.txt
python run_mmlu_open_source.py --model qwen/Qwen-7B --ntrain 0 --batch_size 12 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 8 --streaming --streaming_gap 40 --left 0.10 > zeroshot_qwen_outlierQ8b_0.10.txt
python run_mmlu_open_source.py --model qwen/Qwen-7B --ntrain 0 --batch_size 12 --compress_method densesparseuniformquantization --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 40 --left 0.10 > zeroshot_qwen_outlierQ4b_0.10.txt


python run_mmlu_open_source.py --model qwen/Qwen-7B --ntrain 0 --batch_size 12 --compress_method outquantize_with_lrap --attention_number 40 --quantize_bit 6 --streaming --streaming_gap 40 --left 0.05  --rank 0.05 --rankv 0.05 --loop 3 > zeroshot_qwen_outlierQ6b_0.05_lrap_0.05.txt
python run_mmlu_open_source.py --model qwen/Qwen-7B --ntrain 0 --batch_size 12 --compress_method outquantize_with_lrap --attention_number 40 --quantize_bit 8 --streaming --streaming_gap 40 --left 0.05 --rank 0.05 --rankv 0.05 --loop 3 > zeroshot_qwen_outlierQ8b_0.05_lrap_0.05.txt
python run_mmlu_open_source.py --model qwen/Qwen-7B --ntrain 0 --batch_size 12 --compress_method outquantize_with_lrap --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 40 --left 0.05  --rank 0.05 --rankv 0.05 --loop 3 > zeroshot_qwen_outlierQ4b_0.05_lrap_0.05.txt


