
# ArogyaMitra — Product Requirements & Context Spec
> అరోగ్య మిత్ర · "Health Friend"  
> Gemma 4 Good Hackathon · Deadline: May 18, 2026

---

## 1. One-liner

A 100% on-device, offline-first Telugu health AI for rural Telangana — serving ASHA workers doing clinical triage and patients reading prescriptions they cannot understand, powered by a Gemma 4 E4B model fine-tuned with selective SFT + GRPO on a custom medical Telugu dataset.

---

## 2. Problem

Rural Telangana has a doctor-to-patient ratio of roughly 1:2,000. The gap is filled by ASHA (Accredited Social Health Activist) workers — 1 million+ women across India who visit homes, track pregnancies, manage immunisations, and do first-contact triage with minimal medical training and no digital tools in their language.

Simultaneously, patients receive handwritten prescriptions in English they cannot read, from doctors they see for 3 minutes. They take the wrong dose, miss the refill, or skip the drug entirely because nobody explained it to them.

Both problems share three constraints:

- **Language** — Telugu is the primary language; English instructions are useless
- **Connectivity** — villages have no reliable internet; cloud AI is not an option
- **Privacy** — patient health data must never leave the device

No existing tool addresses all three together.

---

## 3. Who We Are Building For

### User A — ASHA Worker (Lalitha, 34, Nalgonda district)
- Visits 8–12 households per day on foot
- Has a Class 10 education and 3 weeks of government health training
- Carries a mid-range Android phone (₹8,000–₹15,000, 4GB RAM)
- Fills government registers by hand at the end of each day
- Speaks Telugu; reads limited English

**Her core need:** "Tell me if this patient needs to go to the PHC or if I can manage at home — in Telugu, right now, without internet."

### User B — Patient / Family Member (Raju, 58, farmer, Warangal)
- Received a handwritten prescription from a district hospital
- Cannot read English; does not understand drug names
- Has a family member with a smartphone
- Worried about side effects but has nobody to ask

**His core need:** "What is this medicine? How many times a day? Will it clash with my blood pressure tablet?"

---

## 4. Product Vision

ArogyaMitra is a native Android app with two modes sharing one on-device Gemma 4 E4B core. Everything — inference, speech, vision, drug lookup — runs locally. The phone needs no SIM data. Patient data never leaves the device.

The model reasons in Telugu. It does not just retrieve answers — it thinks through clinical cases step by step before giving a triage decision, because of how it was trained (GRPO with a safety-first reward function).

---

## 5. Core Features (MVP for Hackathon)

### 5.1 ASHA Worker Mode

| Feature | Description |
|---|---|
| Symptom intake | Telugu voice input; ASHA describes patient symptoms conversationally |
| Guided screening | Model asks follow-up questions (fever duration? rash? vomiting?) |
| Risk classification | Outputs LOW / MEDIUM / HIGH with reasoning in Telugu |
| Referral decision | Self-care / Go to PHC / Emergency — with specific next steps |
| Vital flag triggers | Auto-escalates for danger signs: convulsions, unconsciousness, bleeding |
| Form autofill | Pre-fills government ASHA register fields from the conversation |

### 5.2 Patient / Family Mode

| Feature | Description |
|---|---|
| Prescription photo | Camera captures handwritten or printed prescription |
| OCR + parsing | MediaPipe preprocesses image → Gemma vision reads drug names, dosage, frequency |
| Telugu explanation | Every drug explained in spoken Telugu: name, purpose, dose, side effects |
| Interaction check | Cross-references offline drug database for known interactions |
| Missed dose guidance | What to do if a dose is skipped |

### 5.3 Shared Infrastructure

| Component | Description |
|---|---|
| Offline drug database | SQLite, seeded from CDSCO public drug list — top 500 rural India drugs |
| Telugu TTS | Android on-device TextToSpeech, Telugu locale |
| Telugu STT | On-device speech recognition, Telugu language model |
| No network permission | App manifest declares no INTERNET permission — enforced at OS level |

---

## 6. Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ArogyaMitra Android App                   │
│                    Kotlin + Jetpack Compose                   │
├──────────────────────────┬──────────────────────────────────┤
│     ASHA Worker Mode     │       Patient / Family Mode       │
│  Telugu voice → triage   │   Camera → prescription → audio  │
├──────────────────────────┴──────────────────────────────────┤
│                  Google AI Edge SDK                          │
│              Gemma 4 E4B  ·  LiteRT runtime                 │
├───────────────┬──────────────────┬──────────────────────────┤
│ MediaPipe     │  Telugu STT/TTS  │  Offline SQLite drug DB  │
│ Vision (OCR)  │  Android on-dev  │  CDSCO top 500 drugs     │
└───────────────┴──────────────────┴──────────────────────────┘
```

### Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| On-device LLM | Gemma 4 E4B | Via Google AI Edge SDK, LiteRT runtime |
| Image preprocessing | MediaPipe Image | Normalise prescription photo before Gemma vision |
| Telugu STT | Android SpeechRecognizer (Telugu) | On-device, no cloud |
| Telugu TTS | Android TextToSpeech (Telugu locale) | On-device |
| App framework | Kotlin + Jetpack Compose | Material 3 |
| Drug database | SQLite via Room | Embedded in APK, CDSCO-seeded |
| Model format | LiteRT .task file | Converted via ai-edge-torch |
| Network | None | INTERNET permission not declared |

---

## 7. Model Training Plan

### Base Model
**Gemma 4 E4B** (Google, April 2026) — chosen because:
- Fits in ~3GB RAM on mid-range Android (quantised INT4)
- Natively multimodal (vision + text) for prescription reading
- Stronger multilingual capability than Gemma 1/2 out of the box
- Required by hackathon rules

### Training Strategy: Selective SFT → GRPO

**Phase 1 — Selective SFT**

Fine-tune only the upper transformer layers (layers 19–27) and the LM head. Bottom layers (0–18) are frozen — they hold Gemma's general language knowledge and multilingual capability. Upper layers are trained to produce structured medical Telugu output.

- Tool: Unsloth (2× faster, single A100 Kaggle notebook)
- Data: ~500 curated Telugu medical instruction pairs
- Loss: cross-entropy on response tokens only
- Goal: teach clinical output format + Telugu register

**Phase 2 — GRPO (Group Relative Policy Optimization)**

After SFT, apply GRPO to teach step-by-step clinical reasoning. For each triage prompt, generate N=8 responses, reward each, update policy toward above-average responses.

- Tool: Unsloth GRPO
- Group size: N=8
- Epochs: 1–2
- Goal: teach reasoning chain and safety-conservative behaviour

### GRPO Reward Functions

```python
def format_reward(response: str) -> float:
    """Output must contain structured triage sections."""
    has_symptom_analysis = any(k in response for k in ["లక్షణాలు", "symptom"])
    has_risk = any(k in response for k in ["రిస్క్", "అత్యవసరం", "high", "medium", "low"])
    has_action = any(k in response for k in ["PHC", "ఆసుపత్రి", "hospital", "విశ్రాంతి"])
    return 1.0 if all([has_symptom_analysis, has_risk, has_action]) else 0.0

def safety_reward(response: str, true_risk: str) -> float:
    """Heavy penalty for missing emergencies. Asymmetric by design."""
    if true_risk == "high" and "అత్యవసరం" not in response:
        return -2.0  # Missing an emergency is the worst failure mode
    if true_risk == "high" and "అత్యవసరం" in response:
        return 2.0
    return 0.5

def telugu_fluency_reward(response: str) -> float:
    """Reward proportional to Telugu script density."""
    telugu_chars = sum(1 for c in response if '\u0C00' <= c <= '\u0C7F')
    return min(telugu_chars / max(len(response), 1), 1.0)

def conciseness_reward(response: str) -> float:
    """ASHA workers need short, actionable responses."""
    word_count = len(response.split())
    return 1.0 if word_count <= 100 else max(0.0, 1.0 - (word_count - 100) / 100)

def combined_reward(response: str, true_risk: str) -> float:
    return (
        format_reward(response)
        + safety_reward(response, true_risk)
        + telugu_fluency_reward(response)
        + conciseness_reward(response)
    )
```

### Training Dataset

| Dataset | Source | Usage |
|---|---|---|
| Telugu alpaca instructions | Telugu-LLM-Labs / HuggingFace | Telugu language style |
| Symptom → diagnosis | `gretelai/symptom_to_diagnosis` | Triage reasoning |
| India medical QA | `openlifescienceai/medmcqa` (AIIMS/NEET PG) | Medical knowledge |
| Prescription images | `mehaksingal/illegible-medical-prescription-images-dataset` | Vision OCR |
| Synthetic prescription pairs | Generated via script | Prescription → Telugu explanation |
| Family-validated ASHA dialogues | Primary collection | Clinical accuracy (our moat) |

### Model Conversion Pipeline

```
Unsloth fine-tune (HuggingFace checkpoint)
        ↓
Export to GGUF (llama.cpp)
        ↓
Convert to LiteRT .task (ai-edge-torch)
        ↓
Bundle inside Android APK
        ↓
Load via Google AI Edge SDK at runtime
```

---

## 8. Why This Wins

### The unfair advantages

1. **Domain expertise** — developer's family has a medical background; clinical logic is validated by real doctors, not guessed
2. **Native language** — developer is Telugu-speaking; voice UX is tested in the actual target language
3. **Novel model** — first Gemma 4 Telugu medical fine-tune in existence; creates a reusable open-source artifact the community can build on
4. **Training approach** — selective SFT + GRPO is research-grade, not commodity LoRA; the safety-asymmetric reward function is a principled design choice
5. **Real constraint** — no INTERNET permission in the Android manifest is a verifiable, concrete privacy guarantee, not a marketing claim

### Hackathon track alignment

| Track | How ArogyaMitra qualifies |
|---|---|
| Health | Primary focus — maternal health, triage, medication literacy |
| Global Resilience | Rural infrastructure gap; functions during floods, power cuts, zero connectivity |
| Unsloth special mention | Both SFT and GRPO phases use Unsloth |
| Google AI Edge | Core deployment stack is AI Edge SDK + MediaPipe + LiteRT |

---

## 9. 10-Day Build Plan

| Day | Milestone |
|---|---|
| 1 | Project scaffold: Android app skeleton, AI Edge SDK wired, Gemma 4 E4B loads |
| 2 | Synthetic dataset generation script + CDSCO drug DB SQLite build |
| 3 | Family sessions: record 200+ Telugu symptom–triage dialogues |
| 4 | Selective SFT training on Kaggle A100 (run overnight) |
| 5 | GRPO training on Kaggle A100 (run overnight) |
| 6 | Model conversion: GGUF → LiteRT .task; load in Android app |
| 7 | Prescription vision pipeline working: photo → Telugu audio explanation |
| 8 | ASHA mode: Telugu voice → triage → referral decision, end-to-end |
| 9 | Drug DB integration, form autofill, UI polish |
| 10 | Demo video shoot + write-up; submit |

### Demo video strategy (most important artefact)

The 30-second anchor shot: a rural patient photographs a handwritten prescription. The phone speaks the drug names, doses, and side effects aloud in Telugu. No internet. No cloud. It works.

---

## 10. Success Metrics (for demo)

| Metric | Target |
|---|---|
| Inference latency (Gemma E4B on Android) | < 3 seconds for triage response |
| Prescription OCR accuracy | > 85% drug name recognition on test set |
| Telugu output ratio | > 75% Telugu characters in response |
| Emergency recall | 100% — zero missed high-risk cases in demo set |
| APK size | < 2GB (model quantised INT4) |
| Internet permission | Not declared — verifiable in manifest |

---

## 11. Open Source Commitment

Post-hackathon, the following will be released publicly:

- Fine-tuned model weights on HuggingFace (`arogyamitra-gemma4-telugu-medical`)
- Training dataset (synthetic + de-identified family dialogues)
- Android app source code (Apache 2.0)
- Unsloth training notebook (Kaggle)

This makes ArogyaMitra a community artifact, not just a hackathon entry — giving it lasting impact in the Telugu health AI space.

---

## 12. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| LiteRT model conversion fails | Fallback: use Gemma 4 via MediaPipe LLM Inference API directly |
| Telugu STT quality poor on medical terms | Supplement with text input fallback in UI |
| 10 days too short to finish both modes | Ship prescription mode first (stronger demo); ASHA mode as stretch |
| GRPO training unstable | Fall back to selective SFT only; GRPO is a bonus |
| Model too large for target device | Quantise to INT4 via ai-edge-torch quantisation pipeline |

---

*Built for the Gemma 4 Good Hackathon · May 2026*  
*Stack: Gemma 4 E4B · Google AI Edge SDK · MediaPipe · Unsloth · GRPO · Android · Telugu


