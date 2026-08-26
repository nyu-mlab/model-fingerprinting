#!/usr/bin/env python3
"""Figure 1: Model Identity Confusion Matrix (Model Fingerprinting)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json, os, re
from collections import defaultdict

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
    'accent': '#C0392B',
}

RAW_DIR = os.path.expanduser('~/workspace/model-fingerprinting/raw_responses')

# True creator mapping
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

ORGS = ['Meta', 'Google', 'Microsoft', 'Alibaba', 'DeepSeek', 'Mistral', 
        'HuggingFace', 'IBM', 'Stability AI', 'AI2', 'Apple', 'OpenAI', 
        'Community', 'Nanbeige', 'None/Other']

org_patterns = {
    'Meta': r'\bmeta\b|llama',
    'Google': r'\bgoogle\b|gemma|deepmind',
    'Microsoft': r'\bmicrosoft\b|\bphi\b',
    'Alibaba': r'\balibaba\b|alibaba cloud|\bqwen\b',
    'DeepSeek': r'\bdeepseek\b',
    'Mistral': r'\bmistral\b',
    'HuggingFace': r'\bhugging\s*face\b',
    'IBM': r'\bibm\b|granite',
    'Stability AI': r'\bstability\b|stable\s*(lm|diffusion)',
    'AI2': r'\bai2\b|allen\s*institute',
    'Apple': r'\bapple\b|openelm',
    'OpenAI': r'\bopenai\b|\bgpt\b|chatgpt',
    'Community': r'\bcommunity\b|open.source',
    'Nanbeige': r'\bnanbeige\b',
}

# Build confusion matrix
models = sorted(TRUE_CREATORS.keys())
matrix = np.zeros((len(models), len(ORGS)), dtype=float)

for mi, model_name in enumerate(models):
    fpath = os.path.join(RAW_DIR, f'{model_name}.jsonl')
    if not os.path.exists(fpath):
        continue
    rows = []
    with open(fpath) as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line.strip()))
                except:
                    pass
    
    identity_rows = [r for r in rows if r.get('category') == 'identity']
    if not identity_rows:
        continue
    
    for r in identity_rows:
        resp = r.get('response', '').lower()
        claimed = []
        for org, pat in org_patterns.items():
            if re.search(pat, resp, re.I):
                claimed.append(org)
        if not claimed:
            claimed = ['None/Other']
        for c in claimed:
            oi = ORGS.index(c)
            matrix[mi, oi] += 1
    
    # Normalize per model
    row_sum = matrix[mi].sum()
    if row_sum > 0:
        matrix[mi] /= row_sum

fig, ax = plt.subplots(figsize=(10, 8))

# Custom colormap: white for 0, blue for correct, red for wrong
im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

ax.set_xticks(range(len(ORGS)))
ax.set_xticklabels(ORGS, rotation=45, ha='right', fontsize=7.5, color=COLORS['text'])
ax.set_yticks(range(len(models)))

# Annotate true creator with checkmark
ylabels = []
for m in models:
    tc = TRUE_CREATORS[m]
    ylabels.append(f'{m} ({tc})')
ax.set_yticklabels(ylabels, fontsize=7, color=COLORS['text'])

ax.set_xlabel('Claimed Creator', fontsize=10, color=COLORS['text'], labelpad=8)
ax.set_ylabel('Model (True Creator)', fontsize=10, color=COLORS['text'], labelpad=8)
ax.set_title('Model Identity Confusion Matrix:\nWho Do Models Claim Created Them?', 
             fontsize=12, fontweight='bold', color=COLORS['text'], pad=12)

# Annotate cells with values
for i in range(len(models)):
    for j in range(len(ORGS)):
        val = matrix[i, j]
        if val > 0.01:
            true_org = TRUE_CREATORS[models[i]]
            is_correct = (ORGS[j] == true_org)
            color = 'white' if val > 0.5 else COLORS['text']
            weight = 'bold' if is_correct else 'normal'
            marker = '✓' if is_correct and val > 0.3 else ''
            ax.text(j, i, f'{marker}{val:.0%}' if val > 0.05 else '', 
                    ha='center', va='center', fontsize=6, color=color, fontweight=weight)

# Draw diagonal highlights for correct matches
for i, m in enumerate(models):
    true_org = TRUE_CREATORS[m]
    if true_org in ORGS:
        j = ORGS.index(true_org)
        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, 
                                    edgecolor=COLORS['accent'], linewidth=1.5, linestyle='--'))

plt.colorbar(im, ax=ax, label='Proportion of Identity Responses', shrink=0.6, pad=0.02)

# Accuracy stat
correct = 0
total = 0
for i, m in enumerate(models):
    true_org = TRUE_CREATORS[m]
    if true_org in ORGS:
        j = ORGS.index(true_org)
        correct += matrix[i, j]
    total += 1

fig.text(0.5, -0.02, 
    f'Fig. 1. Identity confusion matrix for 19 open-weight language models probed with "Who created you?"\n'
    f'Red outlines mark the correct creator. Only {correct/total*100:.0f}% of models correctly self-identify.\n'
    f'Key misattributions: OLMo-2 (AI2) claims OpenAI; SmolLM2 (HuggingFace) claims IBM.',
    ha='center', fontsize=7.5, color=COLORS['light_text'], linespacing=1.4)

plt.savefig('./fig1_identity_confusion.png', 
            dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.15)
plt.savefig('./fig1_identity_confusion.svg', 
            bbox_inches='tight', facecolor='white', pad_inches=0.15)
print("fig1_identity_confusion saved.")
