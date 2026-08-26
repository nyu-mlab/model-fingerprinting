# Model Fingerprinting: Expansion Plan

**Date:** 2026-08-24
**Project:** Model Fingerprinting as MITRE ATLAS Reconnaissance

---

## Current State

### Completed Models (8 of 14 in ZOO)

Each fully probed model has 80 data points (16 probes × 5 trials).

| Model | Family | Size | Developer | Status |
|-------|--------|------|-----------|--------|
| qwen2.5-0.5b | qwen2.5 | 0.5B | Alibaba | **Complete** (80 rows) |
| gemma2-2b | gemma2 | 2.0B | Google | **Complete** (80 rows) |
| granite3.1-2b | granite3.1 | 2.0B | IBM | **Complete** (80 rows) |
| llama3.2-3b | llama3.2 | 3.0B | Meta | **Complete** (80 rows) |
| phi3.5-mini | phi3.5 | 3.8B | Microsoft | **Complete** (80 rows) |
| smollm2-1.7b | smollm2 | 1.7B | HuggingFace | **Complete** (80 rows) |
| stablelm2-1.6b | stablelm2 | 1.6B | Stability AI | **Complete** (80 rows) |
| tinyllama-1.1b | tinyllama | 1.1B | Community | **Complete** (80 rows) |

**Total valid data:** 640 data points across 8 models, 7 families.

### Not Yet Probed (5 in ZOO, never run)

| Model | Family | Size | Developer | Issue |
|-------|--------|------|-----------|-------|
| qwen2.5-1.5b | qwen2.5 | 1.5B | Alibaba | Never probed |
| qwen2.5-3b | qwen2.5 | 3.0B | Alibaba | Never probed |
| llama3.2-1b | llama3.2 | 1.0B | Meta | Never probed |
| smollm2-360m | smollm2 | 360M | HuggingFace | Never probed |
| mistral-7b | mistral | 7.0B | Mistral | Never probed (may OOM in 6GB RAM) |

These are critical for hypotheses H2 (version/size leakage within families) since they provide the multi-size pairs: Qwen 0.5B/1.5B/3B, Llama 1B/3B, SmolLM2 360M/1.7B.

### Dropped

| Model | Reason |
|-------|--------|
| **falcon3-1b** | **DROPPED** per project decision. 6 lines captured (2 of 16 probes, all on id_direct). Probe failures, possibly OOM during import. Not worth further troubleshooting. |

---

## Expansion Plan

### Phase 1: Complete existing ZOO (Priority: HIGH)

Finish the 5 unprobed models already in the design. This is the most important step because the multi-size pairs (Qwen 0.5B/1.5B/3B, Llama 1B/3B, SmolLM2 360M/1.7B) are the core data for **H2 (version/size leakage)**.

- **qwen2.5-1.5b** and **qwen2.5-3b**: Complete the Qwen family trio
- **llama3.2-1b**: Complete the Llama family pair
- **smollm2-360m**: Complete the SmolLM2 family pair (q8_0 quant since repo only ships that)
- **mistral-7b**: May need special handling (tight in 6GB RAM with q4_K_M). If OOM, drop to a smaller Mistral or accept it as the large-model upper bound

**Expected yield:** 400 additional data points (5 × 80), bringing total to 1,040. Families go from 7 to 9 (adding mistral).

### Phase 2: New model families (Priority: HIGH)

Add models from families NOT already represented. Each new family is a new classification class for H1 (family leakage). Prioritized by (a) architectural diversity, (b) MITRE ATLAS relevance, and (c) feasibility on 2 CPU / 6GB RAM.

| Model | Family | Size | Developer | Rationale | GGUF Source |
|-------|--------|------|-----------|-----------|-------------|
| **deepseek-r1-distill-1.5b** | deepseek-r1 | 1.5B | DeepSeek | Reasoning model (RL-trained). Fundamentally different training paradigm from all others. Should exhibit distinctive refusal/reasoning patterns. High ATLAS relevance: attacker fingerprinting a reasoning model vs a standard chat model. | bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF |
| **olmo2-1b** | olmo2 | 1.0B | AI2 | Fully open model (data, code, recipes all public). Tests whether full transparency affects fingerprint (defenders can study it freely, does it still leak?). Different training data mix. | allenai/OLMo-2-0425-1B-Instruct-GGUF |
| **rwkv7-1.5b** | rwkv7 | 1.5B | RWKV Foundation | **Non-transformer architecture** (RNN, O(1) memory, no KV cache). The only non-transformer in the study. Tests whether architecture differences dominate text-based fingerprints. Timing profile should be very different from transformers. | Various community GGUF conversions |
| **openelm-1.1b** | openelm | 1.1B | Apple | Layer-wise scaling architecture. Apple's only small open model. Adds a major developer not currently represented. | apple/OpenELM-1.1B-Instruct-GGUF |
| **nanbeige4.2-3b** | nanbeige | 3.0B | Nanbeige | Chinese-origin model, adds geographic/language training diversity. Tests whether training data language mix leaks through English-only probes. Trending on HuggingFace (18K+ downloads). | Community GGUF |

**Expected yield:** 400 additional data points (5 × 80), bringing total to 1,440. Families go from 9 to 14.

### Phase 3: Cross-generational pairs (Priority: MEDIUM)

Add newer-generation models from families already in the zoo. Tests a new dimension: can fingerprinting detect not just family but *generation* within a family?

| Model | Family | Size | Developer | Rationale |
|-------|--------|------|-----------|-----------|
| **qwen3-0.6b** | qwen3 | 0.6B | Alibaba | Qwen 3 vs Qwen 2.5 already in zoo. Same developer, new generation, 36T tokens training. Hybrid thinking mode. |
| **phi-4-mini** | phi4 | 3.8B | Microsoft | Phi-4 vs Phi-3.5 already in zoo. Same developer, same size, new generation. Direct ablation on generation. |
| **gemma4-e2b** | gemma4 | 2.0B | Google | Gemma 4 vs Gemma 2 already in zoo. Encoder-free architecture change. |

**Expected yield:** 240 additional data points (3 × 80), bringing total to 1,680. Tests a new hypothesis:

**H5 (generation leakage):** Within a developer's model family, different generations (e.g., Qwen 2.5 vs Qwen 3) are distinguishable through black-box probes, even when parameter counts are similar.

### Phase 4: Specialty models (Priority: LOW)

If resources allow, add models that fill specific ATLAS-relevant niches.

| Model | Rationale |
|-------|-----------|
| **antares-1b** (fdtn-ai) | Security-focused model with GraniteMoE architecture. Tests whether security-tuned alignment produces distinctive fingerprints. |
| **falcon3-3b** (TII) | Replaces dropped falcon3-1b with a larger model that may run more reliably. Keeps TII as a represented developer. |

---

## Methodology Adjustments

### No changes to probe battery
The 16-probe battery is well-designed and maps cleanly to ATLAS techniques. Adding probes mid-study would make cross-model comparison harder. Keep the same 16 probes × 5 trials for all new models.

### GGUF availability check
Before adding any model, verify GGUF q4_K_M is available on HuggingFace. Fall back to q8_0 only if necessary (as with smollm2-360m). Document the quantization used per model.

### RAM management
The run_all.py script already handles OOM gracefully (memory gating, one model at a time, crash isolation). Models above 3B in q4_K_M may be tight in 6GB. The mistral-7b is the riskiest. Monitor free_mb() and skip if below floor.

### RWKV special handling
RWKV uses a different inference engine than transformer models under Ollama. Verify that Ollama can serve RWKV-7 GGUF models. If not, may need a separate inference path (rwkv.cpp). This is the only model requiring special infrastructure and should be tested first before committing to the full probe run.

---

## Updated Model Count

| Phase | Models | Families | Data Points |
|-------|--------|----------|-------------|
| Current | 8 | 7 | 640 |
| Phase 1 (complete ZOO) | +5 | +2 | +400 = 1,040 |
| Phase 2 (new families) | +5 | +5 | +400 = 1,440 |
| Phase 3 (cross-gen) | +3 | +3 | +240 = 1,680 |
| Phase 4 (specialty) | +2 | +1 | +160 = 1,840 |
| **Total** | **23** | **18** | **1,840** |

Minimum viable expansion: Phase 1 + Phase 2 = 18 models, 14 families, 1,440 data points. This gives a strong paper with diverse families, a non-transformer control, a reasoning model, and cross-generation pairs for the existing Qwen multi-size family.

---

## Script Changes Required

1. **Update ZOO list in `run_all.py`**: Remove falcon3-1b, add new models with correct HuggingFace repos and GGUF filenames
2. **Update DESIGN.md**: Reflect expanded zoo, add H5 hypothesis if Phase 3 proceeds
3. **Verify GGUF URLs**: Some community conversions may have different naming conventions
4. **RWKV compatibility test**: Run a single probe on RWKV-7 via Ollama to confirm it works before full battery
