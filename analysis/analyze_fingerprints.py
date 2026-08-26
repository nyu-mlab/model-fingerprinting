#!/usr/bin/env python3
"""
Reproduce all key findings from the model fingerprinting raw response data.
Reads raw_responses/*.jsonl and outputs a structured summary.

Usage:
    python3 analyze_fingerprints.py [--raw-dir ../raw_responses]
"""
import json, os, re, argparse
from collections import defaultdict
from pathlib import Path

# True developer labels for each model
TRUE_CREATORS = {
    'llama3.2-1b': 'Meta AI', 'llama3.2-3b': 'Meta AI',
    'qwen2.5-0.5b': 'Alibaba', 'qwen2.5-1.5b': 'Alibaba',
    'qwen2.5-3b': 'Alibaba', 'qwen3-0.6b': 'Alibaba',
    'gemma2-2b': 'Google', 'phi3.5-mini': 'Microsoft',
    'phi4-mini': 'Microsoft', 'deepseek-r1-1.5b': 'DeepSeek',
    'mistral-7b': 'Mistral AI', 'smollm2-1.7b': 'Hugging Face',
    'smollm2-360m': 'Hugging Face', 'granite3.1-2b': 'IBM',
    'stablelm2-1.6b': 'Stability AI', 'olmo2-1b': 'AI2',
    'openelm-1.1b': 'Apple', 'tinyllama-1.1b': 'Community',
    'nanbeige4.2-3b': 'Nanbeige',
}

REFUSAL_KEYWORDS = [
    'sorry', 'cannot', "can't", 'unable', 'i apologize',
    'as an ai', 'i am not able', 'i must decline',
    'not appropriate', 'i refuse', "i'm not able"
]

CREATOR_SEARCH_TERMS = [
    'meta', 'google', 'openai', 'microsoft', 'alibaba',
    'deepseek', 'hugging face', 'huggingface', 'stability ai',
    'ibm', 'apple', 'mistral', 'ai2', 'nanbeige'
]


def load_all_models(raw_dir):
    """Load all JSONL response files into a dict of model_name -> [rows]."""
    models = {}
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith('.jsonl'):
            continue
        model_name = fname.replace('.jsonl', '')
        rows = []
        with open(os.path.join(raw_dir, fname)) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        models[model_name] = rows
    return models


def analyze_identity(models):
    """H1: Identity leakage analysis. Do models correctly self-identify?"""
    print("=" * 60)
    print("FINDING 1: IDENTITY PROBE ANALYSIS")
    print("=" * 60)

    correct = 0
    total = 0
    misidentifications = {}

    for name, rows in sorted(models.items()):
        identity_rows = [r for r in rows if r.get('category') == 'identity']
        true_org = TRUE_CREATORS.get(name, 'Unknown')

        for r in identity_rows:
            resp = r.get('response', '').lower()
            total += 1
            if true_org.lower() in resp:
                correct += 1
            else:
                claims = []
                for org in CREATOR_SEARCH_TERMS:
                    if org in resp:
                        claims.append(org)
                if name not in misidentifications:
                    misidentifications[name] = {
                        'true_creator': true_org,
                        'claims': set()
                    }
                misidentifications[name]['claims'].update(claims)

    failure_rate = (total - correct) / total * 100 if total else 0
    print(f"\nCorrect self-identification: {correct}/{total} ({100-failure_rate:.1f}%)")
    print(f"Identity failure rate: {failure_rate:.1f}%")
    print(f"\nMis-identifications:")
    for name, info in sorted(misidentifications.items()):
        claims = info['claims'] or {'none/unclear'}
        print(f"  {name} (true: {info['true_creator']}) -> claims: {claims}")

    return failure_rate


def analyze_refusals(models):
    """Refusal rate analysis by model and by prompt category."""
    print("\n" + "=" * 60)
    print("FINDING 2: REFUSAL PATTERNS")
    print("=" * 60)

    # Per-model refusal rates
    print("\nPer-model refusal rates:")
    for name, rows in sorted(models.items()):
        refusals = sum(1 for r in rows
                       if any(kw in r.get('response', '').lower()[:500]
                              for kw in REFUSAL_KEYWORDS))
        rate = refusals / len(rows) * 100 if rows else 0
        bar = '#' * int(rate / 2)
        print(f"  {name:25s} {refusals:3d}/{len(rows):3d} ({rate:5.1f}%) {bar}")

    # Per-category refusal rates
    categories = set()
    for rows in models.values():
        for r in rows:
            categories.add(r.get('category', 'unknown'))

    print("\nPer-category refusal rates:")
    for cat in sorted(categories):
        cat_total = 0
        cat_refusals = 0
        for rows in models.values():
            for r in rows:
                if r.get('category') == cat:
                    cat_total += 1
                    if any(kw in r.get('response', '').lower()[:500]
                           for kw in REFUSAL_KEYWORDS):
                        cat_refusals += 1
        rate = cat_refusals / cat_total * 100 if cat_total else 0
        print(f"  {cat:15s} {cat_refusals:3d}/{cat_total:3d} ({rate:5.1f}%)")


def analyze_response_length(models):
    """Response length distributions by model."""
    print("\n" + "=" * 60)
    print("FINDING 3: RESPONSE LENGTH DISTRIBUTIONS")
    print("=" * 60)

    for name, rows in sorted(models.items()):
        lengths = [len(r.get('response', '')) for r in rows]
        if lengths:
            avg = sum(lengths) / len(lengths)
            mn, mx = min(lengths), max(lengths)
            print(f"  {name:25s} avg={avg:6.0f}  min={mn:5d}  max={mx:5d}")


def analyze_determinism(models):
    """Response determinism: do identical prompts produce identical outputs?"""
    print("\n" + "=" * 60)
    print("FINDING 4: RESPONSE DETERMINISM")
    print("=" * 60)

    for name, rows in sorted(models.items()):
        by_prompt = defaultdict(list)
        for r in rows:
            pid = r.get('probe_id', '')
            by_prompt[pid].append(r.get('response', ''))

        multi = {p: resps for p, resps in by_prompt.items() if len(resps) > 1}
        if multi:
            identical = sum(1 for resps in multi.values()
                           if len(set(resps)) == 1)
            total_multi = len(multi)
            rate = identical / total_multi * 100 if total_multi else 0
            print(f"  {name:25s} {identical:2d}/{total_multi:2d} prompts identical ({rate:.0f}%)")


def analyze_format_fingerprints(models):
    """Format fingerprints: markdown, bullets, numbered lists."""
    print("\n" + "=" * 60)
    print("FINDING 5: FORMAT FINGERPRINTS")
    print("=" * 60)

    print(f"\n  {'Model':25s} {'Markdown':>10s} {'Bullets':>10s} {'Numbered':>10s}")
    print("  " + "-" * 60)

    for name, rows in sorted(models.items()):
        fmt_rows = [r for r in rows if r.get('category') == 'format']
        if not fmt_rows:
            continue
        n = len(fmt_rows)
        md = sum(1 for r in fmt_rows
                 if '```' in r.get('response', '') or '**' in r.get('response', ''))
        bullets = sum(1 for r in fmt_rows
                      if re.search(r'^\s*[-*\u2022]', r.get('response', ''), re.M))
        numbered = sum(1 for r in fmt_rows
                       if re.search(r'^\s*\d+\.', r.get('response', ''), re.M))
        print(f"  {name:25s} {md:5d}/{n:<4d} {bullets:5d}/{n:<4d} {numbered:5d}/{n:<4d}")


def analyze_family_clustering(models):
    """Family-level behavioral clustering."""
    print("\n" + "=" * 60)
    print("FINDING 6: FAMILY CLUSTERING")
    print("=" * 60)

    families = defaultdict(list)
    for name in models:
        # Extract family from model name
        if 'qwen2.5' in name:
            families['Qwen 2.5'].append(name)
        elif 'qwen3' in name:
            families['Qwen 3'].append(name)
        elif 'llama3.2' in name:
            families['Llama 3.2'].append(name)
        elif 'smollm2' in name:
            families['SmolLM2'].append(name)
        elif 'phi' in name:
            families['Phi'].append(name)

    for fam, members in sorted(families.items()):
        if len(members) < 2:
            continue
        print(f"\n  {fam} family ({len(members)} members):")
        for m in sorted(members):
            rows = models[m]
            avg_len = sum(len(r.get('response', '')) for r in rows) / len(rows) if rows else 0
            refusals = sum(1 for r in rows
                          if any(kw in r.get('response', '').lower()[:500]
                                 for kw in REFUSAL_KEYWORDS))
            refusal_rate = refusals / len(rows) * 100 if rows else 0
            print(f"    {m:25s} avg_len={avg_len:6.0f}  refusal_rate={refusal_rate:.1f}%")


def print_summary(models, identity_failure_rate):
    """Print overall summary statistics."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_models = len(models)
    total_rows = sum(len(rows) for rows in models.values())
    complete = sum(1 for rows in models.values() if len(rows) >= 80)
    families = len(set(TRUE_CREATORS.get(m, m) for m in models))

    print(f"\n  Models probed:        {total_models}")
    print(f"  Complete (80 rows):   {complete}")
    print(f"  Developer families:   {families}")
    print(f"  Total data points:    {total_rows}")
    print(f"  Identity failure:     {identity_failure_rate:.1f}%")
    print(f"\n  Probe battery: 16 prompts x 5 trials = 80 data points per model")
    print(f"  Categories: identity, refusal, format, math, code,")
    print(f"              degeneration, template, canary, sampling")


def main():
    parser = argparse.ArgumentParser(description='Analyze LLM fingerprinting data')
    parser.add_argument('--raw-dir', default=str(Path(__file__).parent.parent / 'raw_responses'),
                       help='Directory containing raw JSONL response files')
    args = parser.parse_args()

    if not os.path.isdir(args.raw_dir):
        print(f"Error: {args.raw_dir} not found. Run probe.py first.")
        return

    models = load_all_models(args.raw_dir)
    print(f"Loaded {len(models)} models from {args.raw_dir}\n")

    identity_failure = analyze_identity(models)
    analyze_refusals(models)
    analyze_response_length(models)
    analyze_determinism(models)
    analyze_format_fingerprints(models)
    analyze_family_clustering(models)
    print_summary(models, identity_failure)


if __name__ == '__main__':
    main()
