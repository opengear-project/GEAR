![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)
[![arxiv](http://img.shields.io/badge/arxiv-2310.04562-yellow.svg)](https://arxiv.org/pdf/2403.05527.pdf)
## Todo List.
1. simluated code for gsm8k-5shot, bbh-3shot and aqua-8shot with cot prompt on llama models  ✔️
2. Fused quantization supported for GEAR ✔️
3. More cuda kernel optimization 
4. GEAR supported with lm-harness
5. Combining with other inference algorithm/system
6. wrap up a python package
## GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM #

Official repo for `GEAR: An Efficient Error Reduction Framework for KV Cache Compression in LLM Inference.` `GEAR` is a "plug-and-play" inference only KV compression method.
`GEAR` augments any quantization scheme(e.g. KIVI, KCVT and Flexgen) via an error recovery solution to boost the model accuracy while saving memory.

Here, `GEAR` is the abbreviation of `Ge`nerative Inference with LLM via `A`pproximation and Error `R`ecovery.

## Overview
GEAR is an efficient KV cache compression framework that achieves
near-lossless high-ratio compression. GEAR first applies quantization to majority of entries of
similar magnitudes to ultra-low precision. It then employs a low-rank matrix to approximate
the quantization error, and a sparse matrix to remedy individual errors from outlier entries.

GEAR does not need to preserve any first or last tokens uncompressed like other low bit compression algorithms to achieve near lossless KV cache compression for LLMs.
<p align="center"><img width="100%" src="./Fig/overview.png"></p><br/>

## How to use GEAR
```bash
conda create -n GEAR python==3.10
conda activate GEAR
pip install -r requirements.txt
```

### Reposity architecture
```
.
├── GenerationBench
```
`cuda_supported_gear` GEAR-KIVI implementation with fused kernel supported.

`GenerationBench` is simluated compression tested on finetuned and un finetuned model with BBH, GSM8K, and Aqua dataset.

## Developers

- [Hao Kang*](https://haokang-timmy.github.io/)(Georgia Tech)
- [Qingru Zhang*](https://www.linkedin.com/in/qingru-zhang-4b789a187/)(Georgia Tech)
- [Souvik Kundu](https://ksouvik52.github.io/)(Intel)
- [Geonhwa Jeong](https://ghjeong12.github.io/)(Georgia Tech)
- [Zaoxing Liu](https://zaoxing.github.io/)(University of Maryland)
- [Tushar Krishna](https://www.linkedin.com/in/tushar-krishna-a60b0970/)(Georgia Tech)
- [Tuo Zhao](https://www2.isye.gatech.edu/~tzhao80/)(Georgia Tech)


## Citation
Version 2 will be updated soon. Currently it is version 1.
[link to paper](https://arxiv.org/pdf/2403.05527.pdf)
```
@misc{kang2024gear,
      title={GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM}, 
      author={Hao Kang and Qingru Zhang and Souvik Kundu and Geonhwa Jeong and Zaoxing Liu and Tushar Krishna and Tuo Zhao},
      year={2024},
      eprint={2403.05527},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
## Contributing
We are welcoming everyone to contribute to this reposity by rasing PRs. If there is any problem you may also shot email to hkang342@gatech.edu.

## Disclaimer
This “research quality code” is for Non-Commercial purposes and provided by the contributors “As Is” without any express or implied warranty of any kind. The organizations (Intel or georgia Tech) involved do not own the rights to this data set and do not confer any rights to it. The organizations (Intel or georgia Tech) do not warrant or assume responsibility for the accuracy or completeness of any information, text, graphics, links or other items within the code. A thorough security review has not been performed on this code. Additionally, this repository may contain components that are out of date or contain known security vulnerabilities.
