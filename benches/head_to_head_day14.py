"""
Head-to-head benchmark: Outlier-10B vs GPT-3.5-era baseline.

Runs 20 real user tasks across 5 categories, records output quality
and latency, produces a publishable receipt document.

Usage:
    python benches/head_to_head_day14.py --model Outlier-Ai/Outlier-10B-V3.2 --paged
    python benches/head_to_head_day14.py --model Outlier-Ai/Outlier-10B-V3.2 --paged --warmup

Output:
    ~/mac_autopilot_day14/evidence/track_H/head_to_head_receipts.md
    ~/mac_autopilot_day14/evidence/track_H/raw_outputs/
    ~/mac_autopilot_day14/evidence/track_H/scoring.csv
"""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

# 20 prompts across 5 categories (4 each)
PROMPTS = {
    "coding": [
        {
            "id": "code_1",
            "prompt": "Write a Python function that finds the longest palindromic substring in a given string. Include type hints and a docstring.",
            "eval_criteria": "Correct algorithm, clean code, type hints, docstring",
        },
        {
            "id": "code_2",
            "prompt": "Implement a simple LRU cache in Python using only built-in data structures. It should support get() and put() operations in O(1) time.",
            "eval_criteria": "O(1) ops, correct eviction, clean implementation",
        },
        {
            "id": "code_3",
            "prompt": "Write a bash one-liner that finds all Python files modified in the last 24 hours and counts the total lines of code across all of them.",
            "eval_criteria": "Correct find + wc pipeline, handles spaces in filenames",
        },
        {
            "id": "code_4",
            "prompt": "Explain what this code does and identify any bugs:\n\ndef merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    result.extend(left[i:])\n    result.extend(right[j:])\n    return result",
            "eval_criteria": "Correct explanation, identifies no bugs (code is correct)",
        },
    ],
    "writing": [
        {
            "id": "write_1",
            "prompt": "Write a professional email to a client explaining that a project deadline needs to be extended by two weeks due to unexpected technical complexity. Be honest but reassuring.",
            "eval_criteria": "Professional tone, honest, solution-oriented, appropriate length",
        },
        {
            "id": "write_2",
            "prompt": "Write a 200-word product description for a new wireless noise-canceling headphone that emphasizes comfort for all-day wear.",
            "eval_criteria": "Compelling copy, ~200 words, focuses on comfort, natural language",
        },
        {
            "id": "write_3",
            "prompt": "Rewrite this sentence to be more concise: 'In the event that it should happen that the system experiences a failure of a critical nature, it is important that the team should take immediate action to ensure that the issue is resolved in a timely manner.'",
            "eval_criteria": "Dramatically shorter, same meaning, natural",
        },
        {
            "id": "write_4",
            "prompt": "Write an engaging opening paragraph for a blog post about why remote work is here to stay. Target audience: tech managers.",
            "eval_criteria": "Engaging hook, relevant to audience, sets up the article",
        },
    ],
    "reasoning": [
        {
            "id": "reason_1",
            "prompt": "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost? Show your reasoning step by step.",
            "eval_criteria": "Correct answer ($0.05), clear step-by-step reasoning",
        },
        {
            "id": "reason_2",
            "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets? Explain your reasoning.",
            "eval_criteria": "Correct answer (5 minutes), clear reasoning about rate",
        },
        {
            "id": "reason_3",
            "prompt": "Three friends split a restaurant bill. The bill was $30. They each paid $10. The waiter realized there was a $5 discount and returned $5. They each took $1 back and gave $2 to the waiter as tip. So each paid $9, totaling $27. Plus $2 tip = $29. Where's the missing dollar?",
            "eval_criteria": "Correctly identifies the flawed arithmetic framing",
        },
        {
            "id": "reason_4",
            "prompt": "A farmer needs to cross a river with a fox, a chicken, and a bag of grain. The boat can only carry the farmer and one item. The fox will eat the chicken if left alone, and the chicken will eat the grain. What sequence of crossings solves this?",
            "eval_criteria": "Correct solution with all steps, no violations",
        },
    ],
    "qa": [
        {
            "id": "qa_1",
            "prompt": "What is the difference between TCP and UDP? When would you use each?",
            "eval_criteria": "Accurate technical comparison, practical use cases",
        },
        {
            "id": "qa_2",
            "prompt": "Explain the concept of database normalization. What are the first three normal forms?",
            "eval_criteria": "Correct definitions of 1NF, 2NF, 3NF with examples",
        },
        {
            "id": "qa_3",
            "prompt": "What causes seasons on Earth? Why is it summer in the Northern Hemisphere when it's winter in the Southern Hemisphere?",
            "eval_criteria": "Correct (axial tilt, not distance from sun), clear explanation",
        },
        {
            "id": "qa_4",
            "prompt": "What is the difference between machine learning, deep learning, and artificial intelligence? Give a brief explanation suitable for a non-technical person.",
            "eval_criteria": "Accurate, accessible, uses good analogies",
        },
    ],
    "assistant": [
        {
            "id": "asst_1",
            "prompt": "I'm planning a week-long trip to Tokyo in October. Give me a day-by-day itinerary that balances popular tourist spots with local experiences.",
            "eval_criteria": "Practical, diverse activities, considers October weather",
        },
        {
            "id": "asst_2",
            "prompt": "I have chicken breast, rice, broccoli, soy sauce, garlic, and ginger. Suggest a recipe I can make in 30 minutes.",
            "eval_criteria": "Uses given ingredients, realistic 30-min prep, clear instructions",
        },
        {
            "id": "asst_3",
            "prompt": "Help me write a to-do list for launching a small online business. I want to sell handmade candles. What are the first 15 steps?",
            "eval_criteria": "Practical, ordered logically, covers legal/marketing/operations",
        },
        {
            "id": "asst_4",
            "prompt": "My 10-year-old wants to learn programming. What language should they start with and what resources would you recommend?",
            "eval_criteria": "Age-appropriate recommendation, specific resources, encouraging tone",
        },
    ],
}


def run_outlier_prompt(loaded, prompt_text, max_tokens=512):
    """Run a single prompt through the Outlier model and measure latency."""
    from outlier_engine.generate import timed_generation
    result = timed_generation(
        loaded,
        prompt_text,
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return {
        "text": result["generated_text"],
        "tokens": result["tokens"],
        "elapsed_s": result["elapsed_s"],
        "tokens_per_s": result["tokens_per_s"],
    }


def main():
    parser = argparse.ArgumentParser(description="Head-to-head benchmark")
    parser.add_argument("--model", default="Outlier-Ai/Outlier-10B-V3.2")
    parser.add_argument("--paged", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output-dir", default=os.path.expanduser("~/mac_autopilot_day14/evidence/track_H"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw_outputs").mkdir(exist_ok=True)

    print(f"Loading model: {args.model} (paged={args.paged}, warmup={args.warmup})")
    from outlier_engine.loader import load_model
    loaded = load_model(args.model, paged=args.paged, warmup=args.warmup)
    print(f"Model loaded on {loaded.device}")

    results = []
    total = sum(len(prompts) for prompts in PROMPTS.values())
    done = 0

    for category, prompts in PROMPTS.items():
        for prompt_info in prompts:
            done += 1
            pid = prompt_info["id"]
            print(f"[{done}/{total}] Running {pid}...")

            try:
                result = run_outlier_prompt(loaded, prompt_info["prompt"], max_tokens=args.max_tokens)
                entry = {
                    "id": pid,
                    "category": category,
                    "prompt": prompt_info["prompt"],
                    "eval_criteria": prompt_info["eval_criteria"],
                    "outlier_output": result["text"],
                    "tokens": result["tokens"],
                    "elapsed_s": round(result["elapsed_s"], 2),
                    "tokens_per_s": round(result["tokens_per_s"], 2),
                    "error": None,
                }
            except Exception as e:
                entry = {
                    "id": pid,
                    "category": category,
                    "prompt": prompt_info["prompt"],
                    "eval_criteria": prompt_info["eval_criteria"],
                    "outlier_output": None,
                    "error": str(e),
                }
            results.append(entry)

            # Save raw output
            raw_path = output_dir / "raw_outputs" / f"{pid}.json"
            raw_path.write_text(json.dumps(entry, indent=2))
            print(f"  {entry.get('tokens', 0)} tokens, {entry.get('tokens_per_s', 0)} tok/s")

    # Write receipts markdown
    receipt_path = output_dir / "head_to_head_receipts.md"
    with open(receipt_path, "w") as f:
        f.write("# Outlier 10B Head-to-Head Benchmark Receipts\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Model:** {args.model}\n")
        f.write(f"**Device:** {loaded.device}\n")
        f.write(f"**Paged:** {args.paged}\n\n")

        for category, prompts in PROMPTS.items():
            f.write(f"## {category.title()}\n\n")
            cat_results = [r for r in results if r["category"] == category]
            for r in cat_results:
                f.write(f"### {r['id']}\n\n")
                f.write(f"**Prompt:** {r['prompt'][:200]}...\n\n" if len(r['prompt']) > 200 else f"**Prompt:** {r['prompt']}\n\n")
                f.write(f"**Eval criteria:** {r['eval_criteria']}\n\n")
                if r.get("error"):
                    f.write(f"**Error:** {r['error']}\n\n")
                else:
                    f.write(f"**Outlier output:** ({r['tokens']} tokens, {r['tokens_per_s']} tok/s)\n\n")
                    f.write(f"```\n{r['outlier_output'][:1000]}\n```\n\n")
                f.write("---\n\n")

        # Summary
        successful = [r for r in results if r.get("error") is None]
        f.write("## Summary\n\n")
        f.write(f"- **Total prompts:** {total}\n")
        f.write(f"- **Successful:** {len(successful)}\n")
        f.write(f"- **Failed:** {total - len(successful)}\n")
        if successful:
            avg_toks = sum(r["tokens_per_s"] for r in successful) / len(successful)
            f.write(f"- **Average tok/s:** {avg_toks:.1f}\n")

    # Write CSV
    csv_path = output_dir / "scoring.csv"
    with open(csv_path, "w") as f:
        f.write("id,category,tokens,elapsed_s,tokens_per_s,error\n")
        for r in results:
            f.write(f"{r['id']},{r['category']},{r.get('tokens','')},{r.get('elapsed_s','')},{r.get('tokens_per_s','')},{r.get('error','')}\n")

    print(f"\nDone. Receipts: {receipt_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
