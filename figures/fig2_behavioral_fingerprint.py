#!/usr/bin/env python3
"""Figure 2: Behavioral Fingerprint Heatmap Across 19 Models."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json, os, re

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 8,
    'axes.linewidth': 0.5,
    'figure.dpi': 300,
})

COLORS = {
    'text': '#2C3E50',
    'light_text': '#7F8C8D',
}

RAW_DIR = os.path.expanduser('~/workspace/model-fingerprinting/raw_responses')

TRUE_CREATORS = {
    'llama3.2-1b': 'Meta', 'llama3.2-3b': 'Meta',
    'qwen2.5-0.5b': 'Alibaba', 'qwen2.5-1.5b': 'Alibaba', 'qwen2.5-3b': 'Alibaba', 'qwen3-0.6b': 'Alibaba',
    'gemma2-2b': 'Google', 'phi3.5-mini': 'Microsoft', 'phi4-mini': 'Microsoft',
    'deepseek-r1-1.5b': 'DeepSeek', 'mistral-7b': 'Mistral',
    'smollm2-1.7b': 'HuggingFace', 'smollm2-360m': 'HuggingFace',
    'granite3.1-2b': 'IBM', 'stablelm2-1.6b': 'Stability AI',
    'olmo2-1b': 'AI2', 'openelm-1.1b': 'Apple',
    'tinyllama-1.1b': 'Community', 'nanbeige4.2-3b': 'Nanbeige',
}

refusal_kw = ['sorry', 'cannot', "can't", 'unable', 'i apologize', 'not appropriate', 'i must decline']

models_data = {}
for fname in sorted(os.listdir(RAW_DIR)):
    if not fname.endswith('.jsonl'):
        continue
    model_name = fname.replace('.jsonl', '')
    rows = []
    with open(os.path.join(RAW_DIR, fname)) as f:
        for line in f:
            if line.strip():
                try: rows.append(json.loads(line.strip()))
                except: pass
    models_data[model_name] = rows

# Group by family for ordering
families = {
    'Qwen': ['qwen2.5-0.5b', 'qwen2.5-1.5b', 'qwen2.5-3b', 'qwen3-0.6b'],
    'Meta': ['llama3.2-1b', 'llama3.2-3b', 'tinyllama-1.1b'],
    'Microsoft': ['phi3.5-mini', 'phi4-mini'],
    'HuggingFace': ['smollm2-360m', 'smollm2-1.7b'],
    'Others': ['deepseek-r1-1.5b', 'gemma2-2b', 'granite3.1-2b', 'mistral-7b', 
               'nanbeige4.2-3b', 'olmo2-1b', 'openelm-1.1b', 'stablelm2-1.6b'],
}

ordered_models = []
family_boundaries = []
for fam, members in families.items():
    for m in members:
        if m in models_data:
            ordered_models.append(m)
    family_boundaries.append(len(ordered_models))

# Compute behavioral dimensions
dimensions = ['Avg Response\nLength', 'Refusal\nRate', 'Markdown\nUsage', 'Bullet\nUsage', 
              'Numbered\nList Usage', 'Determinism\nScore', 'Empty\nRate']

matrix = np.zeros((len(ordered_models), len(dimensions)))

for mi, model in enumerate(ordered_models):
    rows = models_data[model]
    n = len(rows)
    if n == 0:
        continue
    
    # Avg response length (normalized to 0-1 against max ~500)
    lengths = [len(r.get('response', '')) for r in rows]
    matrix[mi, 0] = min(1.0, np.mean(lengths) / 500)
    
    # Refusal rate
    refusals = sum(1 for r in rows if any(k in r.get('response', '').lower()[:400] for k in refusal_kw))
    matrix[mi, 1] = refusals / n
    
    # Markdown usage (on format prompts)
    format_rows = [r for r in rows if r.get('category') == 'format']
    if format_rows:
        matrix[mi, 2] = sum(1 for r in format_rows if '```' in r.get('response', '') or '**' in r.get('response', '')) / len(format_rows)
        matrix[mi, 3] = sum(1 for r in format_rows if re.search(r'^\s*[-*•]', r.get('response', ''), re.M)) / len(format_rows)
        matrix[mi, 4] = sum(1 for r in format_rows if re.search(r'^\s*\d+\.', r.get('response', ''), re.M)) / len(format_rows)
    
    # Determinism (same prompt → same response)
    from collections import defaultdict
    by_prompt = defaultdict(list)
    for r in rows:
        p = r.get('prompt', '')[:50]
        by_prompt[p].append(r.get('response', ''))
    multi = {p: resps for p, resps in by_prompt.items() if len(resps) > 1}
    if multi:
        identical = sum(1 for resps in multi.values() if len(set(resps)) == 1)
        matrix[mi, 5] = identical / len(multi)
    
    # Empty response rate
    empty = sum(1 for r in rows if len(r.get('response', '').strip()) < 3)
    matrix[mi, 6] = empty / n

fig, ax = plt.subplots(figsize=(9, 8))

im = ax.imshow(matrix, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)

ax.set_xticks(range(len(dimensions)))
ax.set_xticklabels(dimensions, fontsize=8, color=COLORS['text'], ha='center')
ax.set_yticks(range(len(ordered_models)))

# Labels with family coloring
ylabels = []
for m in ordered_models:
    creator = TRUE_CREATORS.get(m, '?')
    ylabels.append(f'{m}')
ax.set_yticklabels(ylabels, fontsize=7.5, color=COLORS['text'])

ax.set_title('Behavioral Fingerprint Heatmap: 19 Open-Weight LLMs', 
             fontsize=12, fontweight='bold', color=COLORS['text'], pad=12)

# Annotate cells
for i in range(len(ordered_models)):
    for j in range(len(dimensions)):
        val = matrix[i, j]
        if val > 0.01:
            color = 'white' if val > 0.6 else COLORS['text']
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6.5, color=color)

# Family separators
for boundary in family_boundaries[:-1]:
    ax.axhline(y=boundary - 0.5, color=COLORS['text'], linewidth=0.8, linestyle='-', alpha=0.3)

# Family labels on right
fam_names = list(families.keys())
prev = 0
for fi, boundary in enumerate(family_boundaries):
    if fi < len(fam_names):
        mid = (prev + boundary - 1) / 2
        ax.text(len(dimensions) + 0.3, mid, fam_names[fi], fontsize=7, color=COLORS['light_text'],
                va='center', fontweight='bold', fontstyle='italic')
    prev = boundary

plt.colorbar(im, ax=ax, label='Normalized Score (0-1)', shrink=0.6, pad=0.08)

fig.text(0.5, -0.02, 
    'Fig. 2. Behavioral fingerprint heatmap across seven dimensions for 19 open-weight language models.\n'
    'Models are grouped by creator family. Each dimension is normalized to [0, 1]. Distinct behavioral\n'
    'clusters emerge within families, suggesting that training methodology and alignment leave detectable traces.',
    ha='center', fontsize=7.5, color=COLORS['light_text'], linespacing=1.4)

plt.savefig('./fig2_behavioral_fingerprint.png', 
            dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.15)
plt.savefig('./fig2_behavioral_fingerprint.svg', 
            bbox_inches='tight', facecolor='white', pad_inches=0.15)
print("fig2_behavioral_fingerprint saved.")
