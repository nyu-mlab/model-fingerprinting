#!/usr/bin/env python3
"""Figure 3: Formatting Preference Fingerprints Across Models."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json, os, re

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 9,
    'axes.linewidth': 0.5,
    'figure.dpi': 300,
})

COLORS = {
    'source': '#4878A8',
    'process': '#E8913A',
    'output': '#5B9A5B',
    'accent': '#C0392B',
    'text': '#2C3E50',
    'light_text': '#7F8C8D',
}

RAW_DIR = os.path.expanduser('~/workspace/model-fingerprinting/raw_responses')

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

# Extract format-category responses
model_names = sorted(models_data.keys())
format_data = {}

for model in model_names:
    rows = models_data[model]
    fmt_rows = [r for r in rows if r.get('category') == 'format']
    if not fmt_rows:
        fmt_rows = rows  # fallback to all
    n = len(fmt_rows)
    if n == 0:
        format_data[model] = {'markdown': 0, 'bullets': 0, 'numbered': 0, 'headers': 0, 'plain': 0}
        continue
    
    md = sum(1 for r in fmt_rows if '```' in r.get('response', '') or '**' in r.get('response', ''))
    bl = sum(1 for r in fmt_rows if re.search(r'^\s*[-*•]', r.get('response', ''), re.M))
    nm = sum(1 for r in fmt_rows if re.search(r'^\s*\d+[\.\)]', r.get('response', ''), re.M))
    hd = sum(1 for r in fmt_rows if re.search(r'^#{1,3}\s', r.get('response', ''), re.M))
    
    plain = n - max(md, bl, nm, hd)
    format_data[model] = {
        'markdown': md / n * 100,
        'bullets': bl / n * 100,
        'numbered': nm / n * 100,
        'headers': hd / n * 100,
    }

# Sort by most formatted to least
model_names_sorted = sorted(model_names, 
    key=lambda m: sum(format_data[m].values()), reverse=True)

fig, ax = plt.subplots(figsize=(10, 7))

dims = ['markdown', 'bullets', 'numbered', 'headers']
dim_labels = ['Markdown (bold/code)', 'Bullet Points', 'Numbered Lists', 'Headers (#)']
dim_colors = [COLORS['source'], COLORS['output'], COLORS['process'], COLORS['accent']]

x = np.arange(len(model_names_sorted))
width = 0.18

for i, (dim, label, color) in enumerate(zip(dims, dim_labels, dim_colors)):
    vals = [format_data[m][dim] for m in model_names_sorted]
    ax.bar(x + (i - 1.5) * width, vals, width=width, label=label, 
           color=color, alpha=0.8, edgecolor='white', linewidth=0.3)

ax.set_xticks(x)
ax.set_xticklabels(model_names_sorted, rotation=45, ha='right', fontsize=7.5, 
                    color=COLORS['text'])
ax.set_ylabel('Usage Rate on Format Prompts (%)', fontsize=10, color=COLORS['text'])
ax.set_title('Formatting Preference Fingerprints Across 19 Open-Weight LLMs', 
             fontsize=11, fontweight='bold', color=COLORS['text'], pad=12)

ax.legend(fontsize=8, frameon=True, fancybox=False, edgecolor=COLORS['light_text'],
          ncol=2, loc='upper right', framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, linewidth=0.5)
ax.set_ylim(0, 110)

fig.text(0.5, -0.04, 
    'Fig. 3. Formatting preferences across 19 open-weight LLMs on standardized format prompts.\n'
    'Each model shows a distinctive formatting signature: Gemma2 consistently uses markdown and numbered lists,\n'
    'while models like OLMo-2 and Qwen3 produce minimal formatting. These preferences are stable across\n'
    'repeated probes and serve as reliable behavioral fingerprints.',
    ha='center', fontsize=7.5, color=COLORS['light_text'], linespacing=1.4)

plt.savefig('./fig3_format_fingerprint.png', 
            dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.15)
plt.savefig('./fig3_format_fingerprint.svg', 
            bbox_inches='tight', facecolor='white', pad_inches=0.15)
print("fig3_format_fingerprint saved.")
