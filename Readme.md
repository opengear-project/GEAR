# FlashKVcache
## Motivation
For long prefill and generate token size tasks(text generation, diaglogue generation), KVcaches could be so huge that loading and saving could be time consuming. Thus systematically and compression optimization is in need for LLMs. Also, previous papers use PPL(perplexity) for evaluation of compressed LLMs. PPL could be low even if generation results are unconvincing. So we evaluate our algorithm in generation tasks and with the right metric.
1. Compression algorithms: Low-rank, quantization.
Lower rank/less bits for not important K,V
2. recompute part of the K V caches.(reduce loading bandwidth, preserve important message)
## Todo(11.1.2023->11.7.2023)
1. Using model [Xgen-7b](https://arxiv.org/abs/2309.03450?ref=blog.salesforceairesearch.com) and set the benchmark of some text summarization tasks.
2. Test of Pi4Cache-Q still works at sentence generation tasks