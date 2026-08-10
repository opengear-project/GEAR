"""
Single-prompt generation test using TrueCompression (FP buffer GEAR method).

Reads a prompt from a .txt file, runs GearLlamaForCausalLMNew.generate() with
optional KV-cache compression, and writes the generated text to an output .txt file.

NOTE: TrueCompression currently only supports Llama-architecture models.

Usage:
    python long_text_generation.py \\
        --prompt_file prompts/my_question_1.txt \\
        --output_file outputs/my_generation.txt \\
        --model meta-llama/Llama-3.2-1B \\
        --compress_method GEAR \\
        --compress_mode gear \\
        --quantize_bit 4 \\
        --rank 4 \\
        --loop 3 \\
        --left 0.02 \\
        --sink_tokens 0 \\
        --recency_tokens 0 \\
        --buffer_len 20 \\clear
        --max_new_tokens 256
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from GEARLM import GearLlamaForCausalLMNew, GearMistralForCausalLMNew


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text from a prompt file using TrueCompression (FP buffer GEAR)."
    )

    # ── Input / output ────────────────────────────────────────────────────────
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        help="Path to .txt file containing the prompt.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to .txt file where generated text will be saved.",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HF hub name or local path (Llama architecture only).",
    )
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token.")
    parser.add_argument("--model_max_length", type=int, default=7000, help="Tokenizer max length.")
    parser.add_argument("--max_new_tokens", type=int, default=7000, help="Max tokens to generate.")
    parser.add_argument("--max_length", type=int, default=None, help="Optional total max length cap.")

    # ── Sampling ──────────────────────────────────────────────────────────────
    parser.add_argument("--do_sample", action="store_true", default=False, help="Enable sampling.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling.")
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.3,
        help="Repetition penalty during generation.",
    )

    # ── Prompt formatting ─────────────────────────────────────────────────────
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        default=False,
        help="Wrap prompt with tokenizer.apply_chat_template (for chat models).",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        # default="You are a creative writing assistant. You are given a prompt and you need to generate a creative writing response based on the story line provided in the prompt.",
        default='you are a scientist who is an expert in the filed mentioned in the prompt , you have to generate an elaborate reposnse based on the prompt.',
        help="System message used when --use_chat_template is set.",
    )

    # ── TrueCompression / FP buffer GEAR ──────────────────────────────────────
    parser.add_argument(
        "--compress_method",
        type=str,
        default="None",
        help="'None' disables compression; any other value (e.g. 'GEAR') enables it.",
    )
    parser.add_argument(
        "--compress_mode",
        type=str,
        default="gear",
        help="Compress mode passed to CompressedUnion: gear / uniform / outlier.",
    )
    parser.add_argument("--quantize_bit", type=int, default=4, help="Quantization bits (4 or 8).")
    parser.add_argument(
        "--rank",
        type=float,
        default=0.0,
        help="Low-rank factor for GEAR. 0 = adaptive rank via SVD.",
    )
    parser.add_argument("--loop", type=int, default=3, help="Power iteration loops for GEAR.")
    parser.add_argument("--left", type=float, default=0.02, help="Outlier fraction for GEAR.")
    parser.add_argument(
        "--buffer_len",
        type=int,
        default=20,
        help="Buffer zone length before FPBuffer flushes to compress.",
    )
    parser.add_argument(
        "--sink_tokens",
        type=int,
        default=4,
        help="Number of initial prefill tokens kept in full precision.",
    )
    parser.add_argument(
        "--recency_tokens",
        type=int,
        default=64,
        help="Size of the rolling recency FP window.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Use StreamCompressedCache instead of CompressedCache.",
    )
    parser.add_argument(
        "--streaming_gap",
        type=int,
        default=1,
        help="Recompress interval when --stream is enabled.",
    )

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument("--debug", action="store_true", default=False, help="Drop into ipdb.")
    return parser.parse_args()


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_compress_config(args):
    if args.compress_method == "None":
        return None
    return {
        "compress_mode": args.compress_mode,
        "quantize_bit": args.quantize_bit,
        "rank": args.rank,
        "loop": args.loop,
        "left": args.left,
        "buffer_len": args.buffer_len,
        "stream": args.stream,
        "streaming_gap": args.streaming_gap,
        "sink_tokens": args.sink_tokens,
        "recency_tokens": args.recency_tokens,
    }


def load_model_and_tokenizer(args, device, compress_config):
    hf_token_kwargs = {"token": args.hf_token} if args.hf_token else {}

    model_kwargs = {
        "torch_dtype": torch.float16,
        "cache_dir": "../cache",
        **hf_token_kwargs,
    }
    if device.type == "cuda":
        model_kwargs["device_map"] = "auto"

    config = transformers.AutoConfig.from_pretrained(
        args.model,
        use_flash_attn=False,
        trust_remote_code=True,
        **hf_token_kwargs,
    )

    # Detect model architecture and select appropriate model class
    model_type = config.model_type.lower()
    if "mistral" in model_type:
        ModelClass = GearMistralForCausalLMNew
        model_arch_name = "Mistral"
    elif "llama" in model_type:
        ModelClass = GearLlamaForCausalLMNew
        model_arch_name = "Llama"
    else:
        raise ValueError(
            f"Unsupported model architecture: {model_type}. "
            f"Supported architectures: llama, mistral"
        )

    logging.info("Loading TrueCompression %s model from: %s", model_arch_name, args.model)
    model = ModelClass.from_pretrained(
        args.model,
        config=config,
        compress_config=compress_config,
        **model_kwargs,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        padding_side="left",
        model_max_length=args.model_max_length,
        use_fast=False,
        cache_dir="../cache",
        **hf_token_kwargs,
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def prepare_prompt(args, tokenizer):
    prompt_path = Path(args.prompt_file)
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    raw_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not raw_prompt:
        raise ValueError(f"Prompt file is empty: {prompt_path}")

    if args.use_chat_template:
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": raw_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = raw_prompt

    return raw_prompt, prompt_text


def build_generate_kwargs(args, tokenizer):
    generate_kwargs = {
        "return_dict_in_generate": True,
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.max_length is not None:
        generate_kwargs["max_length"] = args.max_length

    if args.do_sample:
        generate_kwargs.update(
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    else:
        generate_kwargs.update(
            do_sample=False,
            temperature=None,
            top_k=None,
            top_p=None,
        )
    return generate_kwargs


def main():
    args = parse_args()

    if args.debug:
        import ipdb
        ipdb.set_trace()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    device = select_device()
    logging.info("Using device: %s", device)

    compress_config = build_compress_config(args)
    logging.info("compress_config: %s", compress_config)

    # Import rank tracking functions from TrueCompressFunction
    get_rank_distribution = None
    try:
        from GEARLM.TrueCompression.models.TrueCompressFunction import (
            clear_rank_distribution,
            get_rank_distribution as _get_rank_distribution,
        )
        get_rank_distribution = _get_rank_distribution
        # Clear any previous rank distribution data
        clear_rank_distribution()
        logging.info("Initialized rank distribution tracking")
    except ImportError as exc:
        logging.warning(
            "Could not import rank distribution tracking: %s",
            exc,
        )
    except Exception as exc:
        logging.warning(
            "Error initializing rank distribution tracking: %s",
            exc,
        )

    model, tokenizer = load_model_and_tokenizer(args, device, compress_config)
    raw_prompt, prompt_text = prepare_prompt(args, tokenizer)

    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = inputs.to(device)
    logging.info("Input token count: %d", inputs.input_ids.shape[1])

    generate_kwargs = build_generate_kwargs(args, tokenizer)

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)

    generation = tokenizer.decode(
        outputs.sequences[0, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generation, encoding="utf-8")
    output_dir = output_path.parent

    logging.info("Generation complete (%d chars).", len(generation))
    logging.info("Saved to: %s", output_path.resolve())

    print("=" * 60)
    print("PROMPT FILE:", args.prompt_file)
    print("-" * 60)
    # print(raw_prompt)
    # print("=" * 60)
    # print("GENERATION:")
    # print("-" * 60)
    # print(generation)
    # print("=" * 60)
    print(f"Output written to: {output_path.resolve()}")


    # Get rank distribution data
    rank_distribution = []
    if get_rank_distribution is not None:
        try:
            rank_distribution = get_rank_distribution()
            logging.info("Retrieved rank distribution with %d entries", len(rank_distribution))
        except Exception as exc:
            logging.warning("Error retrieving rank distribution: %s", exc)
    else:
        logging.warning("Rank distribution function not available")
    
    if len(rank_distribution) > 0:
        # Save rank distribution as CSV
        try:
            ranks = np.asarray(rank_distribution, dtype=np.int64)
            counts = np.bincount(ranks)
            csv_data = pd.DataFrame({
                'rank': np.arange(len(counts)),
                'count': counts
            })
            csv_path = output_dir / "rank_distribution.csv"
            csv_data.to_csv(csv_path, index=False)
            logging.info("Saved adaptive rank distribution CSV to %s", csv_path)
        except Exception as exc:
            logging.warning("Could not save rank distribution CSV: %s", exc)
        
        # Generate visualization plot
        if plt is not None:
            try:
                ranks = np.asarray(rank_distribution, dtype=np.int64)
                counts = np.bincount(ranks)
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.bar(np.arange(len(counts)), counts, color="skyblue", edgecolor='black', linewidth=0.5)
                ax.set_title("Adaptive Rank Distribution", fontsize=14, fontweight='bold')
                ax.set_xlabel("Rank", fontsize=12)
                ax.set_ylabel("Count", fontsize=12)
                
                # Limit the number of x-ticks to avoid overcrowding
                num_ticks = min(len(counts), 20)  # Show at most 20 ticks
                tick_positions = np.linspace(0, len(counts) - 1, num_ticks, dtype=int)
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_positions, rotation=45, ha='right', fontsize=10)
                
                # Set minor ticks for all positions
                ax.set_xticks(np.arange(len(counts)), minor=True)
                ax.grid(True, which='minor', axis='x', alpha=0.2, linestyle=':')
                ax.grid(True, which='major', axis='y', alpha=0.3)
                
                fig.subplots_adjust(bottom=0.15)
                plot_path = output_dir / "rank_distribution.png"
                fig.savefig(plot_path, dpi=150)
                plt.close(fig)
                logging.info("Saved adaptive rank distribution plot to %s", plot_path)
            except Exception as exc:
                logging.warning("Could not generate rank distribution plot: %s", exc)
        else:
            logging.warning("matplotlib not available; rank distribution plot was not generated")
    else:
        logging.info("No adaptive rank values were recorded; skipping rank distribution plot and CSV")



if __name__ == "__main__":
    main()