"""
Convert Qwen2 100M config to A2D-Qwen2 config with mask token.

This script:
  1. Loads the existing Qwen2 100M config and tokenizer
  2. Adds a <|mask|> special token to the tokenizer
  3. Creates A2DQwen2Config with the updated vocab_size and mask_token_id
  4. Saves the new config and tokenizer to the output directory

Usage:
    python dllm/examples/gsm_infinity/convert_config.py \
        --src_dir /fast/pmayilvahanan/Interplay-LM-Reasoning/model_configs/qwen2_100M \
        --output_dir /fast/pmayilvahanan/Interplay-LM-Reasoning/dllm/model_configs/a2d_qwen2_100M
"""

import argparse
import json
import os
import shutil

import transformers

from dllm.pipelines.a2d.models.qwen2.modeling_qwen2 import A2DQwen2Config


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen2 100M to A2D-Qwen2 with mask token")
    parser.add_argument(
        "--src_dir",
        type=str,
        default="/fast/pmayilvahanan/Interplay-LM-Reasoning/model_configs/qwen2_100M",
        help="Source Qwen2 100M config directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/fast/pmayilvahanan/Interplay-LM-Reasoning/dllm/model_configs/a2d_qwen2_100M",
        help="Output directory for A2D-Qwen2 config",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load source config ---
    src_config = transformers.AutoConfig.from_pretrained(args.src_dir)
    print(f"Source config: model_type={src_config.model_type}, vocab_size={src_config.vocab_size}")

    # --- Load and modify tokenizer first to get the actual mask token id ---
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.src_dir)

    # Add mask token
    mask_token = "<|mask|>"
    num_added = tokenizer.add_special_tokens({"mask_token": mask_token})
    mask_token_id = tokenizer.mask_token_id
    new_vocab_size = max(src_config.vocab_size, mask_token_id + 1)
    print(f"Added {num_added} special token(s): {mask_token} -> id {mask_token_id}")
    print(f"New vocab_size: {new_vocab_size}")

    # --- Create A2D config ---
    cfg_dict = src_config.to_dict()
    # Remove fields that will be auto-set
    for key in ["model_type", "architectures", "_name_or_path", "transformers_version"]:
        cfg_dict.pop(key, None)

    cfg_dict["vocab_size"] = new_vocab_size
    cfg_dict["mask_token_id"] = mask_token_id
    cfg_dict["pad_token_id"] = 1  # [PAD]
    # Disable cache for diffusion models
    cfg_dict["use_cache"] = False

    a2d_config = A2DQwen2Config(**cfg_dict)
    a2d_config.save_pretrained(args.output_dir)
    print(f"Saved A2D config: model_type={a2d_config.model_type}, vocab_size={a2d_config.vocab_size}, mask_token_id={mask_token_id}")

    # Save tokenizer
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved tokenizer to {args.output_dir}")

    # --- Copy manifest.json if it exists ---
    manifest_src = os.path.join(args.src_dir, "manifest.json")
    if os.path.exists(manifest_src):
        shutil.copy2(manifest_src, os.path.join(args.output_dir, "manifest.json"))

    print(f"\nDone! A2D-Qwen2 100M config saved to: {args.output_dir}")
    print(f"  model_type: {a2d_config.model_type}")
    print(f"  vocab_size: {a2d_config.vocab_size}")
    print(f"  mask_token_id: {mask_token_id}")
    print(f"  hidden_size: {a2d_config.hidden_size}")
    print(f"  num_hidden_layers: {a2d_config.num_hidden_layers}")
    print(f"  num_attention_heads: {a2d_config.num_attention_heads}")


if __name__ == "__main__":
    main()

