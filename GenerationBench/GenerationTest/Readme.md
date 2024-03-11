## GenerationTest
Llama-2-7b,Llama-2-13b and Mistral-7b with GEAR test on GSM8K, GSM8K-COT, MMLU, MMLU-COT, and BBH-COT
## Usage
### BBH
`evaluation_bbh_cot.py` test models on BBH-COT
`evaluation_gsm8k.py` test models on GSM8K and GSM8K-COT
`evaluation_mmlu_cot.py` test models on MMLU-COT
`evaluation_mmlu.py` test models on MMLU
```
usage: evaluation_bbh_cot.py [-h] [--model MODEL] [--tasks TASKS [TASKS ...]] [--prompt_file PROMPT_FILE] [--batch_size BATCH_SIZE] [--max_length MAX_LENGTH] [--max_new_tokens MAX_NEW_TOKENS]
                             [--model_max_length MODEL_MAX_LENGTH] [--do_sample] [--temperature TEMPERATURE] [--top_k TOP_K] [--top_p TOP_P] [--zeroshot] [--dataset_split DATASET_SPLIT]
                             [--example_subset EXAMPLE_SUBSET] [--hf_token HF_TOKEN] [--root_output_dir ROOT_OUTPUT_DIR] [--debug]
                             [--compress_method {groupquantization,groupquantization_token,groupquantization_channel,groupquantization_kc_vt,uniformquantization,poweriteration,outlierquantization,quantize_with_lrap,outliterquantize_with_lrap}]
                             [--rank RANK] [--rankv RANKV] [--loop LOOP] [--quantize_bit QUANTIZE_BIT] [--group_num GROUP_NUM] [--top_kprun TOP_KPRUN] [--left LEFT] [--attention_number ATTENTION_NUMBER] [--gpu GPU]
                             [--streaming] [--streaming_gap STREAMING_GAP]

Evaluate MMLU Tasks

options:
  -h, --help            show this help message and exit
  --model MODEL         Model name or path.
  --tasks TASKS [TASKS ...]
                        The evaluation tasks.
  --prompt_file PROMPT_FILE
  --batch_size BATCH_SIZE
                        Batch size.
  --max_length MAX_LENGTH
                        max length
  --max_new_tokens MAX_NEW_TOKENS
                        max generation length
  --model_max_length MODEL_MAX_LENGTH
                        model max input length
  --do_sample           argument for generation
  --temperature TEMPERATURE
                        argument for generation
  --top_k TOP_K         argument for generation
  --top_p TOP_P         argument for generation
  --zeroshot            whether use zeroshot or cot
  --dataset_split DATASET_SPLIT
                        which part of dataset to choose
  --example_subset EXAMPLE_SUBSET
                        which part of dataset to choose
  --hf_token HF_TOKEN
  --root_output_dir ROOT_OUTPUT_DIR
                        Root output dir
  --debug
  --compress_method {groupquantization,groupquantization_token,groupquantization_channel,groupquantization_kc_vt,uniformquantization,poweriteration,outlierquantization,quantize_with_lrap,outliterquantize_with_lrap}
                        choose one of the compression method
  --rank RANK           rank compared with smaller dimension set to K cache.
  --rankv RANKV         rank compared with smaller dimension set to V cache.
  --loop LOOP           loop of SVD solver, default = 0
  --quantize_bit QUANTIZE_BIT
                        Quantize bit of algorithm
  --group_num GROUP_NUM
                        group number of group quantization
  --top_kprun TOP_KPRUN
  --left LEFT           outlier extraction part compared with total matrix
  --attention_number ATTENTION_NUMBER
                        attention layer number of LLM, for LlAMA-2-7b it is 32
  --gpu GPU
  --streaming           Use streaming mode.
  --streaming_gap STREAMING_GAP
                        iteration length for re-compression
```
