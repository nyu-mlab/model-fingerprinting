# Model Fingerprinting as MITRE ATLAS Reconnaissance — Project Design

**Institution:** NYU Tandon School of Engineering | **Started:** 2026-06-25
**Affiliation:** NYU Tandon School of Engineering

---

## 1. Motivation & Gap

The existing LLM-fingerprinting literature is almost entirely framed around **IP
protection / ownership verification** — the *defender* wants to prove "this
deployed model is mine." Representative work:

- **TRAP** (Gubri et al., 2024) — adversarial suffixes force a target answer; black-box identity verification (BBIV). >95% TPR @ <0.2% FPR.
- **LLMmap** (2024) — active fingerprinting with ~8 crafted queries.
- **ZEROPRINT** (2025) — zeroth-order gradient estimation; Fisher-information argument that gradients carry more identity than outputs.
- **Refusal-vector behavioral fingerprint** (2025) — provenance via refusal directions.
- **A Fingerprint for LLMs** (2024) — output vector-space similarity.
- **ESF / FPEdit / Scalable Fingerprinting** — injected/backdoor watermark families.

**The gap:** nobody frames fingerprinting from the **attacker's** point of view
as a *reconnaissance* capability, and nobody maps probe families to a threat
taxonomy. Yet MITRE ATLAS already has the vocabulary:

- **AML.TA0001 Reconnaissance** → *Discover ML Model Ontology*, *Discover ML Artifacts*, *Active Scanning*
- **AML.TA0004 ML Model Access** → *AML.T0040 ML Model Inference API Access*
- **AML.TA0008 Defense Evasion** → *LLM Meta Prompt Extraction*
- **AML.TA0000-ish staging** → recon feeds *Craft Adversarial Data* / *Evade ML Model*

Fingerprinting **is** reconnaissance: knowing the exact model family/version
lets an attacker select transferable jailbreaks, known CVEs, prompt-injection
payloads, and extraction attacks tuned to that model. That reframing is the
contribution.

## 2. Hypothesis (falsifiable)

**H1 (family leakage).** Black-box LLMs leak their *developer family* through
purely API-observable behavior (response text + timing), recoverable by a cheap
probe battery at accuracy far above the class-prior baseline.

**H2 (version/size leakage).** Within a family, *model size/version* is also
recoverable above chance, though harder than family.

**H3 (probe prioritization).** Probe *types* differ markedly in discriminative
power, and can be ranked into an ATLAS-style technique-prioritization table
(analogous to ATT&CK technique prioritization). Identity/refusal/timing probes
will dominate; generic factual probes will be weak.

**H4 (timing side-channel).** Output throughput (tokens/sec) is a system-level
fingerprint that leaks model size independent of text content — a recon side
channel (maps to Active Scanning).

**Null we try to break:** "Black-box text responses are too noisy /
deployment-normalized to identify the model above chance." If probes can't beat
prior, H1 is falsified.

## 3. Experimental Design

### 3.1 Model zoo (ground truth — we own the labels)
Quantized GGUF (q4_K_M unless noted), CPU inference via Ollama. Families chosen
for (a) cross-family diversity and (b) intra-family multi-size for H2.

Multi-size families (H2):
- **Qwen2.5-Instruct** (Alibaba): 0.5B, 1.5B, 3B
- **Llama-3.2-Instruct** (Meta): 1B, 3B
- **SmolLM2-Instruct** (HuggingFace): 360M, 1.7B

Single representatives (H1 breadth):
- **Gemma-2-2b-it** (Google)
- **Phi-3.5-mini-instruct** (Microsoft)
- **TinyLlama-1.1B-Chat** (community/Llama arch)
- **StableLM-2-1.6B-chat** (Stability AI)
- **Granite-3.1-2b-instruct** (IBM)
- **Falcon3-1B-Instruct** (TII)
- **Mistral-7B-Instruct** (Mistral) — the one large model, breadth anchor

Target: ~15 models / ~10 families. Each is a classification class with full
ground-truth labels (family, size, developer, architecture, quant).

### 3.2 Probe battery (the "recon scan")
Each probe = one prompt + decoding config. Probe categories, each tagged to an
ATLAS technique:

| # | Category | Example | ATLAS technique |
|---|----------|---------|-----------------|
| P1 | Direct identity | "What model are you? Who made you?" | Discover ML Model Ontology |
| P2 | Indirect identity | "What's your knowledge cutoff?" | Discover ML Model Ontology |
| P3 | Refusal style | borderline-but-benign request | Discover ML Artifacts (alignment) |
| P4 | Deterministic format | "List the planets." (temp 0) | Active Scanning |
| P5 | Math/format | "Compute 17*23, show work." | Active Scanning |
| P6 | Code style | "Write a Python is_prime." | Active Scanning |
| P7 | Repetition/degeneration | long open-ended @ temp 0 | Active Scanning |
| P8 | Template/special-token leak | "Repeat: <|im_start|>" | LLM Meta Prompt Extraction |
| P9 | Canary/memorization | famous-string completion | Discover ML Artifacts |
| P10 | Sampling distribution | open prompt @ temp>0, many trials | Active Scanning |

### 3.3 Protocol
- Each probe sent to each model **N=5 trials** (temp=0 deterministic probes:
  trials measure timing variance; temp>0 probes: trials measure distribution).
- Fixed seeds where deterministic; record everything.
- Capture: full response text, eval_count (out tokens), prompt_eval_count,
  eval_duration, load/total duration, timestamp.
- **Raw responses are the data of record** → `raw_responses/*.jsonl`. Never
  overwrite; provenance per row (model label, probe id, trial, seed, timing).

### 3.4 Features (black-box only)
Text: length (char/token), self-ID vendor regex hits, refusal flag, markdown
usage (code fences/bullets/bold/headers), emoji, first-token & first-word,
list/enumeration style, punctuation profile, char n-gram TF-IDF, type-token
ratio, exact-match hash (temp 0 collision across models).
Timing: tokens/sec (eval_count/eval_duration), prompt processing rate.

### 3.5 Analysis
- **Unsupervised:** TF-IDF + cosine → hierarchical clustering / t-SNE; do
  responses cluster by family without labels? (purity, ARI).
- **Supervised:** stratified CV classifier (logistic reg / random forest) on
  features → family accuracy (H1), size-within-family accuracy (H2), confusion
  matrices, per-probe ablation (H3).
- **Probe prioritization (H3):** train one-probe-at-a-time; rank by accuracy +
  mutual information → ATLAS-style priority table.
- **Timing (H4):** tokens/sec distributions per size; ANOVA/regression
  size→throughput; can timing alone separate sizes?
- Baselines: majority-class prior, random.

### 3.6 Threats to validity (own them)
- Deployment normalization (system prompts, output filters) can erase text
  signals — we test ours undefended; note this as the realistic upper bound.
- Quantization may alter behavior vs full precision — we fingerprint the
  *deployed artifact*, which is the realistic target; documented.
- Small/quantized models only (compute-bound) — generalization to frontier
  API models is argued, not proven; framed as future work.
- Single inference stack (llama.cpp/Ollama) — timing is stack-relative.

## 4. Deliverables
1. Reproducible harness + probe battery (code).
2. Raw response corpus (JSONL, full provenance).
3. Feature matrix (CSV) + analysis notebooks/scripts.
4. Figures: cluster map, confusion matrices, probe-priority bar, timing plots.
5. Written report (paper-style) with the ATLAS technique-prioritization table —
   the concrete research deliverable.

## 5. Status log
- 2026-06-25: env characterized (2 CPU/6GB/no GPU); Ollama CPU build sideloaded;
  HF-CDN GGUF path established (Ollama R2 blocked); Qwen2.5-0.5B imported &
  inference confirmed (4.4s/resp, self-ID + repetition signal observed).
