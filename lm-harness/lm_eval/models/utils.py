from transformers.models.llama.modeling_llama import LlamaForCausalLM
from typing import List, Literal, Optional, Tuple, Union
import torch
def self_define_model(
    self,
    pretrained: str,
    revision: Optional[str] = "main",
    dtype: Optional[Union[str, torch.dtype]] = "auto",
    trust_remote_code: Optional[bool] = False,
    # arguments used for splitting a model across GPUs naively.
    # only used if `parallelize=True`.
    # (accelerate naive PP (device_map) options)
    parallelize: Optional[bool] = False,
    device_map_option: Optional[str] = "auto",
    max_memory_per_gpu: Optional[Union[int, str]] = None,
    max_cpu_memory: Optional[Union[int, str]] = None,
    offload_folder: Optional[str] = "./offload",
    # PEFT and quantization options
    peft: Optional[str] = None,
    autogptq: Optional[Union[bool, str]] = False,
    **kwargs,
):
    model_kwargs = kwargs if kwargs else {}
    self._model = LlamaForCausalLM.from_pretrained(
        pretrained,
        revision=revision,
        trust_remote_code=trust_remote_code,
        device_map = "auto",
        **model_kwargs,
        
    )
