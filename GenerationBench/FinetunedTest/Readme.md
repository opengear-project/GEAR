## FinetunedTest
Llama-2-7b fintuned with GSM8K tested on GSM8K
## Usage
There is an example of how to run `test_gsm8k` at `run.sh`
```
usage: test_gsm8k.py [-h] [--model_name_or_path MODEL_NAME_OR_PATH] [--adapter_name_or_path ADAPTER_NAME_OR_PATH] [--ckpt_dir CKPT_DIR] [--full_precision [FULL_PRECISION]] [--token TOKEN]
                     [--model_max_length MODEL_MAX_LENGTH] [--device_map DEVICE_MAP] [--data_name DATA_NAME] [--batch_size BATCH_SIZE] [--compress_method COMPRESS_METHOD] [--rank RANK] [--rankv RANKV] [--loop LOOP]
                     [--quantize_bit QUANTIZE_BIT] [--group_num GROUP_NUM] [--left LEFT] [--attention_number ATTENTION_NUMBER] [--gpu GPU] [--streaming [STREAMING]] [--streaming_gap STREAMING_GAP]

options:
  -h, --help            show this help message and exit
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to the model. (default: LoftQ/Mistral-7B-v0.1-4bit-64rank)
  --adapter_name_or_path ADAPTER_NAME_OR_PATH
                        Path to the LoRA adapter. Used in evaluation or resuming from the checkpoint. (default: None)
  --ckpt_dir CKPT_DIR   Path to your local output directory (default: None)
  --full_precision [FULL_PRECISION]
  --token TOKEN         HF token to access to private models, e.g., meta-llama (default: None)
  --model_max_length MODEL_MAX_LENGTH
                        Maximum sequence length. Sequences will be left padded (and possibly truncated). (default: 512)
  --device_map DEVICE_MAP
                        Path to your local output directory (default: cuda:0)
  --data_name DATA_NAME
                        Dataset name. (default: gsm8k)
  --batch_size BATCH_SIZE
                        Evaluation batch size. (default: 16)
  --compress_method COMPRESS_METHOD
                        compression strategy of models. (default: None)
  --rank RANK           rank compared with smaller dimension set to K cache. (default: 0.0)
  --rankv RANKV         rank compared with smaller dimension set to V cache (default: 0.0)
  --loop LOOP           loop of SVD solver, default = 0 (default: 0)
  --quantize_bit QUANTIZE_BIT
                        Quantize bit of algorithm (default: 8)
  --group_num GROUP_NUM
                        group number of group quantization (default: 0)
  --left LEFT           outlier extraction part compared with total matrix (default: 0.0)
  --attention_number ATTENTION_NUMBER
                        attention layer number of LLM, for LlAMA-2-7b it is 32 (default: 100)
  --gpu GPU             set gpu (default: 1)
  --streaming [STREAMING]
                        Use streaming mode. (default: False)
  --streaming_gap STREAMING_GAP
                        iteration length for re-compression (default: 0)
```
