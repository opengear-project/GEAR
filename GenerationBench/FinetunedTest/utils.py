from dataclasses import dataclass, field
from typing import Dict, Optional

import transformers


from datasets import load_dataset


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="LoftQ/Mistral-7B-v0.1-4bit-64rank",
        metadata={"help": "Path to the model."},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the LoRA adapter. Used in evaluation or resuming from the checkpoint."
        },
    )
    ckpt_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to your local output directory"},
    )
    full_precision: bool = field(default=False)
    token: Optional[str] = field(
        default=None,
        metadata={"help": "HF token to access to private models, e.g., meta-llama"},
    )
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be left padded (and possibly truncated)."
        },
    )
    device_map: Optional[str] = field(
        default="cuda:0",
        metadata={"help": "Path to your local output directory"},
    )


@dataclass
class DataArguments:
    data_name: str = field(default="gsm8k", metadata={"help": "Dataset name."})
    batch_size: int = field(default=16, metadata={"help": "Evaluation batch size."})


@dataclass
class CompressArguments:
    compress_method: Optional[str] = field(
        default="None",
        metadata={"help": "compression strategy of models."},
    )
    rank: float = field(
        default=0.0,
        metadata={"help": "rank compared with smaller dimension set to K cache."},
    )
    rankv: float = field(
        default=0.0,
        metadata={"help": "rank compared with smaller dimension set to V cache"},
    )
    loop: int = field(default=0, metadata={"help": "loop of SVD solver, default = 0"})
    quantize_bit: int = field(default=8, metadata={"help": "Quantize bit of algorithm"})
    group_num: int = field(
        default=0, metadata={"help": "group number of group quantization"}
    )
    left: float = field(
        default=0.0,
        metadata={"help": "outlier extraction part compared with total matrix"},
    )
    attention_number: int = field(
        default=100,
        metadata={"help": "attention layer number of LLM, for LlAMA-2-7b it is 32"},
    )
    stage: int = field(default=0, metadata={"help": "Evaluation batch size."})
    gpu: int = field(default=1, metadata={"help": "Evaluation batch size."})
    iter: int = field(default=1, metadata={"help": "Evaluation batch size."})
    locality_saving: float = field(
        default=0.0, metadata={"help": "Evaluation batch size."}
    )
    start_saving: float = field(
        default=0.0, metadata={"help": "Evaluation batch size."}
    )
    token_preserving: bool = field(
        default=False, metadata={"help": "Evaluation batch size."}
    )
    streaming: bool = field(default=False, metadata={"help": "Evaluation batch size."})
    streaming_gap: int = field(default=0, metadata={"help": "Evaluation batch size."})


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
