# Gemma-4-E4B-it Adapter Training

## Base

google/gemma-4-E4B-it

Training:
- QLoRA
- Unsloth
- 4bit
- Same base model for all adapters

---

# Adapter 1 — Telugu

Goal:
- Telugu fluency
- Telugu instruction following
- Telugu generation

Datasets:
- Indic Instruct
- Indic Alpaca
- English↔Telugu Parallel

Output:
adapters/telugu

---

# Adapter 2 — Medical

Goal:
- Medical QA
- Medicine understanding
- Dosage reasoning
- Healthcare responses

Datasets:
- MedMCQA
- MeDiaQA
- WHO docs
- Healthcare FAQs

Output:
adapters/medical

---

# Adapter 3 — OCR Robustness

Goal:
- OCR typo correction
- Prescription normalization
- Medicine correction

Datasets:
- OCR prescription datasets
- Synthetic OCR corruption pairs

Example:
Input:
Paracetmol 650

Output:
Paracetamol 650

Output:
adapters/ocr

---

# Adapter 4 — ASHA

Goal:
- Simplified Telugu healthcare conversations
- Rural-friendly responses
- Safety/escalation responses

Datasets:
- Synthetic ASHA conversations
- Telugu healthcare dialogues

Output:
adapters/asha

---

# Common LoRA Config

r = 16
lora_alpha = 16
lora_dropout = 0

Target Modules:
- q_proj
- k_proj
- v_proj
- o_proj
- gate_proj
- up_proj
- down_proj

---

# Save Adapters Separately

adapters/
├── telugu
├── medical
├── ocr
└── asha

---

# Initial Strategy

Train adapters separately first.

DO NOT merge during training.

Test:
- Telugu
- Medical
- OCR

individually before:
- stacking
- routing
- TIES merging
- DARE merging