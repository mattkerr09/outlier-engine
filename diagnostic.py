import gc
import glob
import json
import os
from collections import OrderedDict

import psutil
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from outlier_engine.paging import OutlierPagedModel, _detect_format, _remap_real_key
from outlier_engine.quantize_utils import quantize_to_int8


def rss():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


def tensor_gb(tensors):
    total = 0
    for t in tensors:
        total += t.nelement() * t.element_size()
    return total / 1024**3

print(f"Baseline: {rss():.2f} GB")

from outlier_engine.loader import load_model, load_tokenizer  # noqa: F401
print(f"After import: {rss():.2f} GB")

model_dir = snapshot_download('Outlier-Ai/Outlier-10B')
print(f"After snapshot_download: {rss():.2f} GB")

with open(os.path.join(model_dir, 'config.json')) as f:
    cfg = json.load(f)
model = OutlierPagedModel(model_dir, device='cpu')
print(f"After full paged load: {rss():.2f} GB")
resident_tensors = []
for p in model.parameters():
    resident_tensors.append(p.detach())
for b in model.buffers():
    resident_tensors.append(b.detach())
print(f"Resident tensor bytes after full paged load: {tensor_gb(resident_tensors):.2f} GB")
print(f"Overhead after full paged load: {rss() - tensor_gb(resident_tensors):.2f} GB")
del resident_tensors

del model
gc.collect()
print(f"After deleting full paged model: {rss():.2f} GB")

shards = sorted(glob.glob(os.path.join(model_dir, '*.safetensors')))
print(f"Shard count: {len(shards)}")
print(f"Before index scan: {rss():.2f} GB")
index = {}
first_keys = []
for s in shards:
    with safe_open(s, framework='pt', device='cpu') as f:
        keys = list(f.keys())
        if not first_keys:
            first_keys = keys
        for k in keys:
            if '.mlp.experts.' in k and 'shared_expert' not in k:
                index[k] = s
print(f"After index scan: {rss():.2f} GB")
fmt = _detect_format(first_keys)
print('Format:', fmt, 'expert_tensors:', len(index))

from outlier_engine.paging import _PagedLayer, _RMSNorm, _NoInitEmbedding, _NoInitLinear
cfg_norm = {
    'n_layers': cfg['num_hidden_layers'],
    'hidden_dim': cfg['hidden_size'],
    'n_heads': cfg['num_attention_heads'],
    'n_kv_heads': cfg['num_key_value_heads'],
    'intermediate_dim': cfg['intermediate_size'],
    'vocab_size': cfg['vocab_size'],
    'max_seq_len': cfg.get('max_position_embeddings', 4096),
    'rope_theta': cfg['rope_parameters']['rope_theta'],
    'n_experts': cfg['n_experts'],
    'top_k': cfg['top_k'],
}
D,I,V=cfg_norm['hidden_dim'],cfg_norm['intermediate_dim'],cfg_norm['vocab_size']
embed = _NoInitEmbedding(V,D)
layers = torch.nn.ModuleList([_PagedLayer(D,I,cfg_norm['n_heads'],cfg_norm['n_kv_heads'],cfg_norm['rope_theta'],cfg_norm['max_seq_len'],cfg_norm['n_experts'],cfg_norm['top_k']) for _ in range(cfg_norm['n_layers'])])
norm = _RMSNorm(D)
lm_head = _NoInitLinear(D,V,bias=False)
# keep float weights fp16
for m in (embed, layers, norm, lm_head):
    m.to(dtype=torch.float16)
print(f"After empty module init: {rss():.2f} GB")

assigned = 0
for shard in shards:
    with safe_open(shard, framework='pt', device='cpu') as f:
        for raw_key in f.keys():
            if '.mlp.experts.' in raw_key and 'shared_expert' not in raw_key:
                continue
            key = _remap_real_key(raw_key)
            t = f.get_tensor(raw_key)
            if key == 'embed_tokens.weight':
                embed.weight.data.copy_(t.to(embed.weight.dtype))
            elif key == 'norm.weight':
                norm.weight.data.copy_(t.to(norm.weight.dtype))
            elif key == 'lm_head.weight':
                lm_head.weight.data.copy_(t.to(lm_head.weight.dtype))
            else:
                import re
                m = re.match(r'^layers\.(\d+)\.(.+)$', key)
                if not m:
                    continue
                li = int(m.group(1)); suffix = m.group(2); layer = layers[li]
                if suffix == 'attn_norm.weight':
                    layer.attn_norm.weight.data.copy_(t.to(layer.attn_norm.weight.dtype))
                elif suffix == 'ffn_norm.weight':
                    layer.ffn_norm.weight.data.copy_(t.to(layer.ffn_norm.weight.dtype))
                elif suffix.startswith('attn.'):
                    proj_name, param_name = suffix[len('attn.'):].rsplit('.',1)
                    mod = getattr(layer.attn, proj_name)
                    getattr(mod, param_name).data.copy_(t.to(getattr(mod, param_name).dtype))
                elif suffix == 'ffn.router_weight':
                    layer.ffn.router_weight.data.copy_(t.to(layer.ffn.router_weight.dtype))
                elif suffix.startswith('ffn.shared.'):
                    shared = layer.ffn.shared
                    proj = suffix[len('ffn.shared.'):]
                    q,s = quantize_to_int8(t)
                    if proj == 'gate_proj.weight':
                        shared.gate_w.copy_(q); shared.gate_s.copy_(s.to(shared.gate_s.dtype))
                    elif proj == 'up_proj.weight':
                        shared.up_w.copy_(q); shared.up_s.copy_(s.to(shared.up_s.dtype))
                    elif proj == 'down_proj.weight':
                        shared.down_w.copy_(q); shared.down_s.copy_(s.to(shared.down_s.dtype))
            del t
            gc.collect()
            assigned += 1
print(f"After streaming shared load + quantize: {rss():.2f} GB")
resident_tensors = []
for obj in [embed, norm, lm_head, *list(layers)]:
    for p in obj.parameters(recurse=True):
        resident_tensors.append(p.detach())
    for b in obj.buffers(recurse=True):
        resident_tensors.append(b.detach())
print(f"Resident tensor bytes in streamed skeleton: {tensor_gb(resident_tensors):.2f} GB")
print(f"Overhead in streamed skeleton: {rss() - tensor_gb(resident_tensors):.2f} GB")
print(f"Assigned tensors: {assigned}")
