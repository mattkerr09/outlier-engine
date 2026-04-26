#!/usr/bin/env python3
"""
Repack Outlier-70B V3.2 HF export into paging-engine-compatible format.

The 70B HF export stores experts as individual files:
  experts/layer_14_expert_00.safetensors  (bare keys: down_ternary, gate_scale, ...)

The paging engine (load_hybrid_paged_qwen) expects:
  model_dir/*.safetensors with keys like:
    base.model.layers.14.mlp.experts.0.down_ternary
    base.model.layers.14.mlp.experts.0.down_scale
    ...

This script:
1. Reads config.json for architecture info
2. Copies Qwen2.5-32B base model shards into the output dir
3. Repacks per-file experts into combined safetensors shards with correct key prefixes
4. Copies config.json, tokenizer, alpha.json, router_state into the output dir
5. Adds n_kv_heads to config.json (paging engine bug workaround)

Usage:
    python repack_70b_for_paging.py \
        --raw-dir checkpoints/outlier-70b-v3.2-raw \
        --base-dir checkpoints/qwen25-32b-base \
        --out-dir checkpoints/outlier-70b-v3.2-paged
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, help="Path to raw 70B HF export")
    parser.add_argument("--base-dir", required=True, help="Path to Qwen2.5-32B-Instruct base model")
    parser.add_argument("--out-dir", required=True, help="Output directory for repacked model")
    parser.add_argument("--experts-per-shard", type=int, default=8,
                        help="Number of experts per output shard (default: 8 = 1 layer)")
    args = parser.parse_args()

    raw = Path(args.raw_dir)
    base = Path(args.base_dir)
    out = Path(args.out_dir)

    # Validate inputs
    if not (raw / "config.json").exists():
        print(f"ERROR: {raw}/config.json not found"); sys.exit(1)
    if not (raw / "experts").is_dir():
        print(f"ERROR: {raw}/experts/ not found"); sys.exit(1)
    if not any(base.glob("*.safetensors")):
        print(f"ERROR: No safetensors in {base}"); sys.exit(1)

    out.mkdir(parents=True, exist_ok=True)

    # 1. Read config
    with open(raw / "config.json") as f:
        cfg = json.load(f)
    n_experts = cfg.get("n_experts") or cfg.get("outlier_num_experts", 8)
    moe_layers = cfg.get("moe_layers", [])
    n_kv_heads = cfg.get("num_key_value_heads")
    print(f"Config: {cfg.get('num_hidden_layers')} layers, {n_experts} experts, "
          f"moe_layers={len(moe_layers)}, n_kv_heads={n_kv_heads}")
    print(f"Expert files: {len(list((raw / 'experts').glob('*.safetensors')))}")

    # 2. Symlink base model shards into output dir (saves ~62 GB disk)
    print("\n[Step 1/4] Symlinking base model shards...")
    base_shards = sorted(base.glob("*.safetensors"))
    for shard in base_shards:
        dst = out / shard.name
        if not dst.exists():
            dst.symlink_to(shard.resolve())
            print(f"  Linked {shard.name} -> {shard.resolve()}")
        else:
            print(f"  Already exists: {shard.name}")

    # 3. Copy config, tokenizer, and supporting files
    print("\n[Step 2/4] Copying config and tokenizer files...")
    for fname in ["tokenizer.json", "tokenizer_config.json", "merges.txt",
                   "vocab.json", "generation_config.json",
                   "alpha.json", "router_state.safetensors", "manifest.json",
                   "training_summary.json"]:
        src = raw / fname
        if src.exists():
            shutil.copy2(src, out / fname)
            print(f"  Copied {fname}")
    # Also check base dir for tokenizer files not in raw
    for fname in ["tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"]:
        dst = out / fname
        src = base / fname
        if not dst.exists() and src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied {fname} from base")

    # 4. Write patched config.json with n_kv_heads fix
    print("\n[Step 3/4] Writing patched config.json...")
    patched_cfg = dict(cfg)
    if "n_kv_heads" not in patched_cfg and n_kv_heads is not None:
        patched_cfg["n_kv_heads"] = n_kv_heads
        print(f"  Added n_kv_heads={n_kv_heads}")
    # Ensure model_type is outlier_moe
    if patched_cfg.get("model_type") != "outlier_moe":
        patched_cfg["model_type"] = "outlier_moe"
        print("  Set model_type=outlier_moe")
    # Remove auto_map (we use load_hybrid_paged_qwen, not AutoModel)
    patched_cfg.pop("auto_map", None)
    # Remove base_model_name_or_path (local cluster path, not useful here)
    patched_cfg.pop("base_model_name_or_path", None)
    with open(out / "config.json", "w") as f:
        json.dump(patched_cfg, f, indent=2)
    print(f"  Wrote config.json")

    # 5. Repack experts into combined shards
    print(f"\n[Step 4/4] Repacking {len(moe_layers)} layers x {n_experts} experts "
          f"({args.experts_per_shard} experts/shard)...")

    # Parse expert files
    expert_pat = re.compile(r"layer_(\d+)_expert_(\d+)\.safetensors")
    expert_files = {}
    for ef in sorted((raw / "experts").glob("*.safetensors")):
        m = expert_pat.match(ef.name)
        if m:
            layer_idx, expert_idx = int(m.group(1)), int(m.group(2))
            expert_files[(layer_idx, expert_idx)] = ef

    print(f"  Found {len(expert_files)} expert files")

    # Group experts into shards
    all_experts = sorted(expert_files.keys())
    shard_groups = []
    current_group = []
    for key in all_experts:
        current_group.append(key)
        if len(current_group) >= args.experts_per_shard:
            shard_groups.append(current_group)
            current_group = []
    if current_group:
        shard_groups.append(current_group)

    # Determine shard numbering (after base model shards)
    existing_shards = sorted(out.glob("model-*.safetensors"))
    next_shard_idx = len(existing_shards)
    total_shards = next_shard_idx + len(shard_groups)

    for group_idx, group in enumerate(shard_groups):
        shard_idx = next_shard_idx + group_idx
        shard_name = f"expert-{shard_idx:05d}-of-{next_shard_idx + len(shard_groups):05d}.safetensors"
        shard_path = out / shard_name

        tensors = {}
        for layer_idx, expert_idx in group:
            ef = expert_files[(layer_idx, expert_idx)]
            data = load_file(str(ef))
            for bare_key, tensor in data.items():
                prefixed_key = f"base.model.layers.{layer_idx}.mlp.experts.{expert_idx}.{bare_key}"
                tensors[prefixed_key] = tensor

        save_file(tensors, str(shard_path))
        layers_in_group = sorted(set(l for l, _ in group))

        # Incremental cleanup: delete raw expert files that were just repacked
        for layer_idx, expert_idx in group:
            ef = expert_files[(layer_idx, expert_idx)]
            ef.unlink(missing_ok=True)

        if group_idx % 5 == 0 or group_idx == len(shard_groups) - 1:
            print(f"  Shard {group_idx + 1}/{len(shard_groups)}: "
                  f"{shard_name} ({len(tensors)} tensors, layers {layers_in_group})")

    # Summary
    all_out_safetensors = sorted(out.glob("*.safetensors"))
    total_size_gb = sum(f.stat().st_size for f in all_out_safetensors) / 1e9
    print(f"\nRepack complete:")
    print(f"  Output dir: {out}")
    print(f"  Base shards: {len(base_shards)}")
    print(f"  Expert shards: {len(shard_groups)}")
    print(f"  Total safetensors: {len(all_out_safetensors)}")
    print(f"  Total size: {total_size_gb:.1f} GB")
    print(f"  Config: {out / 'config.json'}")


if __name__ == "__main__":
    main()
