"""
Create Qwen2.5-based model configs for FineWeb pretraining.

Creates both AR (Qwen2ForCausalLM) and A2D (diffusion) configs at three sizes
(100M, 200M, 400M), all using the full Qwen2.5 tokenizer (~151K vocab).

Usage:
    python scripts/create_fineweb_model_configs.py
"""

import json
import os
import shutil

from transformers import AutoConfig, AutoTokenizer


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

QWEN25_MODEL = "Qwen/Qwen2.5-0.5B"

# Architecture specs — non-embedding params roughly match the GSM-Infinity sizes.
# With a 151K vocab and tie_word_embeddings=true, the embedding table adds
# vocab_size * hidden_size params (shared with lm_head).  The "model name"
# (100M/200M/400M) refers to the *non-embedding* transformer body.
MODEL_SPECS = {
    "100M": dict(
        hidden_size=768,
        num_hidden_layers=12,
        intermediate_size=3072,
        num_attention_heads=12,
        num_key_value_heads=2,
    ),
    "200M": dict(
        hidden_size=768,
        num_hidden_layers=24,
        intermediate_size=3072,
        num_attention_heads=12,
        num_key_value_heads=2,
    ),
    "400M": dict(
        hidden_size=1024,
        num_hidden_layers=26,
        intermediate_size=4096,
        num_attention_heads=16,
        num_key_value_heads=4,
    ),
}


def create_ar_config(size_name: str, spec: dict, tokenizer, ref_config):
    """Create a Qwen2ForCausalLM config for AR pretraining."""
    out_dir = os.path.join(PROJECT_ROOT, "model_configs", f"qwen2_fineweb_{size_name}")
    os.makedirs(out_dir, exist_ok=True)

    cfg = {
        "architectures": ["Qwen2ForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "hidden_act": "silu",
        "hidden_size": spec["hidden_size"],
        "initializer_range": 0.02,
        "intermediate_size": spec["intermediate_size"],
        "max_position_embeddings": 2048,
        "max_window_layers": spec["num_hidden_layers"],
        "model_type": "qwen2",
        "num_attention_heads": spec["num_attention_heads"],
        "num_hidden_layers": spec["num_hidden_layers"],
        "num_key_value_heads": spec["num_key_value_heads"],
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "sliding_window": 2048,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "use_mrope": False,
        "use_sliding_window": False,
        "vocab_size": len(tokenizer),
    }

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=False)

    tokenizer.save_pretrained(out_dir)
    print(f"  AR config: {out_dir}  (vocab={cfg['vocab_size']}, layers={cfg['num_hidden_layers']}, hidden={cfg['hidden_size']})")
    return out_dir


def create_a2d_config(size_name: str, spec: dict, tokenizer, ref_config):
    """Create an A2D-Qwen2 config for diffusion (MDLM/BD3LM) pretraining."""
    out_dir = os.path.join(PROJECT_ROOT, "dllm", "model_configs", f"a2d_qwen2_fineweb_{size_name}")
    os.makedirs(out_dir, exist_ok=True)

    mask_token = "<|mask|>"
    tok = tokenizer.__class__.from_pretrained(tokenizer.name_or_path)
    tok.add_special_tokens({"mask_token": mask_token})
    mask_token_id = tok.mask_token_id
    new_vocab_size = max(len(tok), mask_token_id + 1)

    n_layers = spec["num_hidden_layers"]

    cfg = {
        "attention_dropout": 0.0,
        "bos_token_id": tokenizer.bos_token_id,
        "dtype": "bfloat16",
        "eos_token_id": tokenizer.eos_token_id,
        "hidden_act": "silu",
        "hidden_size": spec["hidden_size"],
        "initializer_range": 0.02,
        "intermediate_size": spec["intermediate_size"],
        "layer_types": ["full_attention"] * n_layers,
        "mask_token_id": mask_token_id,
        "max_position_embeddings": 2048,
        "max_window_layers": n_layers,
        "model_type": "a2d-qwen2",
        "num_attention_heads": spec["num_attention_heads"],
        "num_hidden_layers": n_layers,
        "num_key_value_heads": spec["num_key_value_heads"],
        "pad_token_id": tokenizer.eos_token_id,
        "rms_norm_eps": 1e-6,
        "rope_scaling": None,
        "rope_theta": 1000000.0,
        "sliding_window": None,
        "tie_word_embeddings": True,
        "use_cache": False,
        "use_mrope": False,
        "use_sliding_window": False,
        "vocab_size": new_vocab_size,
    }

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=False)

    tok.save_pretrained(out_dir)
    print(f"  A2D config: {out_dir}  (vocab={new_vocab_size}, mask_id={mask_token_id}, layers={n_layers}, hidden={cfg['hidden_size']})")
    return out_dir


def main():
    print(f"Downloading reference config and tokenizer from {QWEN25_MODEL}...")
    ref_config = AutoConfig.from_pretrained(QWEN25_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(QWEN25_MODEL)
    print(f"  Reference vocab size: {len(tokenizer)}")
    print(f"  Reference config: hidden={ref_config.hidden_size}, layers={ref_config.num_hidden_layers}")
    print()

    for size_name, spec in MODEL_SPECS.items():
        print(f"--- {size_name} ---")
        create_ar_config(size_name, spec, tokenizer, ref_config)
        create_a2d_config(size_name, spec, tokenizer, ref_config)
        print()

    print("Done. All FineWeb model configs created.")


if __name__ == "__main__":
    main()
