# LLM Model Fingerprinting as MITRE ATLAS Reconnaissance

Behavioral probing of open-weight language models for supply chain verification. Maps LLM fingerprinting techniques to the [MITRE ATLAS](https://atlas.mitre.org/) threat taxonomy, reframing model identification as an adversarial reconnaissance capability rather than a defensive IP-protection tool.

**Institution:** NYU Tandon School of Engineering

## Key Findings (Preliminary)

From 19 models across 14 developer families, probed with a 16-prompt battery (80 data points per model, 1,491 total):

- **87% identity failure rate.** Only 13% of models correctly identify their own developer when asked directly. OLMo-2 (AI2) claims OpenAI. SmolLM2 (Hugging Face) claims IBM. Self-reported identity is unreliable for supply chain verification.
- **Distinctive format fingerprints.** Formatting preferences (markdown, numbered lists, bullet points) vary consistently by model family and are stable across prompts. Gemma-2 always uses markdown + numbered lists; OLMo-2 uses no formatting at all.
- **High response determinism.** 80-100% of same-prompt responses are character-identical at temperature 0, making behavioral fingerprinting reproducible and stable.
- **Refusal patterns cluster by family.** Identity probes trigger refusals in Gemma, Nanbeige, and Qwen but not in Llama or Phi. Different safety alignment strategies leave distinct behavioral traces.
- **Size scaling effects.** Within the Qwen 2.5 family, refusals decrease and responses shorten as model size increases (0.5B → 1.5B → 3B).

## Model Zoo

| Model | Developer | Size | Status |
|-------|-----------|------|--------|
| deepseek-r1-1.5b | DeepSeek | 1.5B | Complete (80 rows) |
| gemma2-2b | Google | 2.0B | Complete (80 rows) |
| granite3.1-2b | IBM | 2.0B | Complete (80 rows) |
| llama3.2-1b | Meta AI | 1.0B | Complete (80 rows) |
| llama3.2-3b | Meta AI | 3.0B | Complete (80 rows) |
| mistral-7b | Mistral AI | 7.0B | In progress (51 rows) |
| nanbeige4.2-3b | Nanbeige | 3.0B | Complete (80 rows) |
| olmo2-1b | AI2 | 1.0B | Complete (80 rows) |
| openelm-1.1b | Apple | 1.1B | Complete (80 rows) |
| phi3.5-mini | Microsoft | 3.8B | Complete (80 rows) |
| phi4-mini | Microsoft | 3.8B | Complete (80 rows) |
| qwen2.5-0.5b | Alibaba | 0.5B | Complete (80 rows) |
| qwen2.5-1.5b | Alibaba | 1.5B | Complete (80 rows) |
| qwen2.5-3b | Alibaba | 3.0B | Complete (80 rows) |
| qwen3-0.6b | Alibaba | 0.6B | Complete (80 rows) |
| smollm2-1.7b | Hugging Face | 1.7B | Complete (80 rows) |
| smollm2-360m | Hugging Face | 360M | Complete (80 rows) |
| stablelm2-1.6b | Stability AI | 1.6B | Complete (80 rows) |
| tinyllama-1.1b | Community | 1.1B | Complete (80 rows) |

All models run as quantized GGUF (Q4_K_M unless noted) via [Ollama](https://ollama.com/) on CPU.

## Repository Structure

```
model-fingerprinting/
├── README.md                    # This file
├── DESIGN.md                    # Research design, hypotheses, methodology
├── expansion-plan.md            # Model zoo expansion roadmap
├── scripts/
│   ├── probe.py                 # Probe battery (16 prompts x 5 trials)
│   ├── download_zoo.py          # GGUF model download script
│   ├── orchestrate.py           # Multi-model probe orchestrator
│   ├── run_all.py               # Full zoo probe runner
│   └── probe_remaining.py       # Resume probing incomplete models
├── analysis/
│   └── analyze_fingerprints.py  # Reproduces all key findings from raw data
├── raw_responses/               # JSONL files (one per model, 80 rows each)
│   ├── deepseek-r1-1.5b.jsonl
│   ├── gemma2-2b.jsonl
│   └── ...                      # 19 model response files
├── figures/
│   ├── fig1_identity_confusion.{png,svg,py}
│   ├── fig2_behavioral_fingerprint.{png,svg,py}
│   └── fig3_format_fingerprint.{png,svg,py}
└── models/
    ├── modelfiles/              # Ollama Modelfile configs per model
    └── zoo_manifest.json        # Model zoo metadata
```

## Reproduction

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- GGUF model files (downloaded via `scripts/download_zoo.py`)

### Run probes

```bash
# Probe a single model
python3 scripts/probe.py <model-name> --trials 5

# Resume an interrupted probe run
python3 scripts/probe.py <model-name> --resume

# Run all models in the zoo
python3 scripts/run_all.py
```

### Analyze results

```bash
# Reproduce all key findings from raw response data
python3 analysis/analyze_fingerprints.py
```

### Generate figures

```bash
# Each figure has a standalone generation script
python3 figures/fig1_identity_confusion.py
python3 figures/fig2_behavioral_fingerprint.py
python3 figures/fig3_format_fingerprint.py
```

## Probe Battery

16 probes across 9 categories, each mapped to a MITRE ATLAS technique:

| Category | Probes | ATLAS Technique |
|----------|--------|-----------------|
| Identity | 3 | Discover ML Model Ontology |
| Refusal | 2 | Discover ML Artifacts |
| Format | 2 | Active Scanning |
| Math | 2 | Active Scanning |
| Code | 1 | Active Scanning |
| Degeneration | 1 | Active Scanning |
| Template leak | 2 | LLM Meta Prompt Extraction |
| Canary | 1 | Discover ML Artifacts |
| Sampling | 2 | Active Scanning |

## Related Work

- TRAP (Gubri et al., 2024): adversarial suffix fingerprinting
- LLMmap (2024): active fingerprinting with crafted queries
- ZEROPRINT (2025): zeroth-order gradient fingerprinting
- Refusal-vector behavioral fingerprint (2025)

This work differs by framing fingerprinting as **attacker reconnaissance** rather than defender IP protection, and by mapping probe families to the MITRE ATLAS threat taxonomy.

## License

MIT

## Citation

Part of ongoing research at NYU Tandon School of Engineering.
