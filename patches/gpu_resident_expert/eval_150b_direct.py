"""Direct 150B V3.2 MMLU eval bypassing lm_eval's .to() OOM."""
import json, time, sys, os
os.environ["HF_HOME"] = "/mnt/1tbfin3/hf_cache"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

print("Loading 150B model...", flush=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    "/mnt/1tbfin3/base_150b",
    trust_remote_code=True,
    dtype=torch.bfloat16,
)
print(f"Model loaded in {time.time()-t0:.0f}s", flush=True)
for i in range(torch.cuda.device_count()):
    print(f"  GPU{i}: {torch.cuda.memory_allocated(i)/1024**3:.1f}GB", flush=True)

tokenizer = AutoTokenizer.from_pretrained("/mnt/1tbfin3/base_150b", trust_remote_code=True)

# Create HFLM wrapper WITHOUT calling .to()
lm = HFLM.__new__(HFLM)
lm._model = model
lm.tokenizer = tokenizer
lm._device = torch.device("cuda:0")
lm.batch_size_per_gpu = 1
lm._max_length = 2048
lm.add_bos_token = False
lm.logits_cache = True
lm.truncation = False
lm._batch_size = 1
lm.batch_schedule = 1
lm._rank = 0
lm._world_size = 1
lm.custom_prefix_token_id = None
lm.prefix_token_id = tokenizer.bos_token_id
lm.revision = "main"
lm.delta = None
lm.peft = None
lm._config = model.config

print("Running MMLU eval...", flush=True)
results = evaluator.simple_evaluate(
    model=lm,
    tasks=["mmlu"],
    batch_size=1,
    log_samples=False,
)

# Save results
out_path = "/workspace/outlier-eval/results/150b_v3_2_mmlu_full.json"
with open(out_path, "w") as f:
    json.dump(results["results"], f, indent=2, default=str)

mmlu = results["results"].get("mmlu", {})
acc = mmlu.get("acc,none", mmlu.get("acc"))
stderr = mmlu.get("acc_stderr,none", mmlu.get("acc_stderr"))
print(f"\n150B V3.2 MMLU: acc={acc} stderr={stderr}", flush=True)
print(f"Saved to {out_path}", flush=True)
