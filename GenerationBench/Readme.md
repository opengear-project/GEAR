## Generation Bench
All results here are tested by simulated compression code.

## Reposity architecture
```
.
├── FinetunedTest
├── GenerationTest
```
`FinetunedTest` is using Llama-2-7b finetuned on GSM8K and test GEAR on GSM8K. Thanks for [Yixiao Li](https://scholar.google.com/citations?user=KZIAP7MAAAAJ&hl=en) who shared the model for us.

`GenerationTest` is using Llama-2-7b,Llama-2-13b and Mistral-7b with GEAR test on GSM8K, GSM8K-COT, MMLU, MMLU-COT, and BBH-COT.
