# python3 main.py --dataset "kmfoda/booksum" --batch-size 6 --maxlength 4000 --max-new-tokens 128 > llama2_booksum_128.txt
# python3 main.py --dataset "kmfoda/booksum" --batch-size 6 --maxlength 4000 --max-new-tokens 192 > llama2_booksum_192.txt
python3 main.py --dataset "kmfoda/booksum" --batch-size 6 --maxlength 4000 --max-new-tokens 256 > llama2_booksum_256.txt
python3 main.py --dataset "kmfoda/booksum" --batch-size 6 --maxlength 4000 --max-new-tokens 320 > llama2_booksum_320.txt
python3 main.py --dataset "kmfoda/booksum" --batch-size 6 --maxlength 4000 --max-new-tokens 384 > llama2_booksum_384.txt