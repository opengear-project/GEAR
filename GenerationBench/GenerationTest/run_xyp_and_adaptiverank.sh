# Activate the virtual environment
source /home/cse240d-fal26-lora/GEAR/cse240venv/bin/activate

# python evaluation_gsm8k_true_compression.py \
#   --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
#   --prompt_file gsm8k_prompt_original.txt \
#   # --example_subset 0:10 \
#   --compress_method GEAR \
#   --compress_mode gear \
#   --batch_size 8 \
#   --quantize_bit 4 \
#   --rank 1 \
#   --loop 3 \
#   --left 0.02 \
#   --sink_tokens 4 \
#   --recency_tokens 64 \
#   --buffer_len 20 \
#   --max_new_tokens 256


# python evaluation_gsm8k_true_compression.py \
#   --model meta-llama/Meta-Llama-3-8B \
#   --prompt_file gsm8k_prompt_original.txt \
#   # --example_subset 0:10 \
#   --compress_method GEAR \
#   --compress_mode gear \
#   --batch_size 6 \
#   --quantize_bit 2 \
#   --rank 16 \
#   --loop 3 \
#   --left 0.02 \
#   --sink_tokens 16 \
#   --recency_tokens 64 \
#   --buffer_len 20 \
#   --max_new_tokens 256


# python evaluation_aqua_cot_true_compression.py \
#   --model meta-llama/Meta-Llama-3-8B \
#   --compress_method GEAR \
#   --compress_mode gear \
#   --batch_size 1 \
#   --quantize_bit 2 \
#   --rank 16 \
#   --loop 3 \
#   --left 0.02 \
#   --sink_tokens 16 \
#   --recency_tokens 64 \
#   --buffer_len 20 \
#   --max_new_tokens 256


# python evaluation_bbh_cot_true_compression.py \
#   --model meta-llama/Meta-Llama-3-8B \
#   --task multistep_arithmetic_two \
#   # --example_subset 0:10 \
#   --compress_method GEAR \
#   --compress_mode gear \
#   --batch_size 6 \
#   --quantize_bit 4 \
#   --rank 16 \
#   --loop 3 \
#   --left 0.02 \
#   --sink_tokens 16 \
#   --recency_tokens 64 \
#   --buffer_len 20 \
#   --max_new_tokens 256




##############################################################################################################

  #coherent text generation test
#   cd GenerationBench/GenerationTest

# python long_text_generation.py \
#   --prompt_file prompts/my_question_2.txt \
#   --output_file outputs/my_generation.txt \
#   --model mistralai/Mistral-7B-Instruct-v0.3 \
#   --use_chat_template \
#   --compress_method GEAR \
#   --compress_mode gear \
#   --quantize_bit 4 \
#   --rank 4 \
#   --loop 3 \
#   --left 0.02 \
#   --sink_tokens 0 \
#   --recency_tokens 0 \
#   --buffer_len 20 \
#   --max_new_tokens 5000


# python scrolls_test.py \
#   --scrolls_subset gov_report \
#   --example_subset 0:2 \
#   --max_new_tokens 512 \
#   --model_max_length 4096 \
#   --compress_method None



##############################################################################################################
# Mistral tests


# python evaluation_gsm8k_true_compression.py \
#   --model mistralai/Mistral-7B-Instruct-v0.3 \
#   --prompt_file gsm8k_prompt_original.txt \
#   # --example_subset 0:2 \
#   --compress_method GEAR \
#   --compress_mode gear \
#   --batch_size 6 \
#   --quantize_bit 2 \
#   --rank 4 \
#   --loop 3 \
#   --left 0.02 \
#   --sink_tokens 16 \
#   --recency_tokens 32 \
#   --buffer_len 20 \
#   --max_new_tokens 256


# python evaluation_aqua_cot_true_compression.py \
#   --model mistralai/Mistral-7B-Instruct-v0.3 \
#   # --example_subset 0:2 \
#   --compress_method GEAR \
#   --compress_mode gear \
#   --batch_size 6 \
#   --quantize_bit 2 \
#   --rank 4 \
#   --loop 3 \
#   --left 0.02 \
#   --sink_tokens 16 \
#   --recency_tokens 32 \
#   --buffer_len 20 \
#   --max_new_tokens 256


python evaluation_bbh_cot_true_compression.py \
  --model meta-llama/Meta-Llama-3-8B \
  --compress_method GEAR \
  --task reasoning_about_colored_objects \
  --compress_mode gear \
  --batch_size 1 \
  --quantize_bit 2 \
  --sink_tokens 4 \
  --recency_tokens 32 \
  --buffer_len 20 \
  --max_new_tokens 256