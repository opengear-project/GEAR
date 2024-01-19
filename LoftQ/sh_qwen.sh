python3 test_gsm8k.py --model_name_or_path Qwen/Qwen-7B --batch_size 8 --device_map "cuda:0" --gpu 0 > benchmark_untune_qwen.txt

python3 test_gsm8k.py --model_name_or_path Qwen/Qwen-7B --batch_size 8 --device_map "cuda:0" --gpu 0 --compress_method uniformquantization --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 40 > qwen7b_uniQ4.txt
python3 test_gsm8k.py --model_name_or_path Qwen/Qwen-7B --batch_size 8 --device_map "cuda:0" --gpu 0 --compress_method uniformquantization --attention_number 40 --quantize_bit 6 --streaming --streaming_gap 40 > qwen7b_uniQ6.txt
python3 test_gsm8k.py --model_name_or_path Qwen/Qwen-7B --batch_size 8 --device_map "cuda:0" --gpu 0 --compress_method uniformquantization --attention_number 40 --quantize_bit 8 --streaming --streaming_gap 40 > qwen7b_uniQ8.txt


python3 test_gsm8k.py --model_name_or_path Qwen/Qwen-7B --batch_size 8 --device_map "cuda:0" --gpu 0 --compress_method uniformquantization --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 40 > qwen7b_uniQ4.txt
python3 test_gsm8k.py --model_name_or_path Qwen/Qwen-7B --batch_size 8 --device_map "cuda:0" --gpu 0 --compress_method uniformquantization --attention_number 40 --quantize_bit 6 --streaming --streaming_gap 40 > qwen7b_uniQ6.txt
python3 test_gsm8k.py --model_name_or_path Qwen/Qwen-7B --batch_size 8 --device_map "cuda:0" --gpu 0 --compress_method uniformquantization --attention_number 40 --quantize_bit 8 --streaming --streaming_gap 40 > qwen7b_uniQ8.txt