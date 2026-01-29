# Codes for a DP-based LLM split inference framework based on Pangu-Embedded-7B and Ascend 910B

## Download

```python
python download.py
```
---

## Training

### Step 1: Train the Encoder–Decoder

Run:

```bash
bash train_proj.sh
```

This script trains the encoder–decoder model used for split inference.

---

### Step 2: Search for Privacy Parameter

1. Perform a **binary search** over the DP mechanism parameter **μ** to reach the target attack success rate (ASR).
2. For the selected **μ**, train the following script:

```bash
bash train_soft.sh
```

---

## Testing

### Evaluate Coherence Between Generated Text and Ground Truth

Run:

```bash
bash test_cse.sh
```

---

## InferDPT Baseline

### Step 1: Export Tokenizer Vocabulary

Enter the `inferdpt` directory:

```bash
cd inferdpt
python export_vocab.py
```

---

### Step 2: Search for ε and Run Inference Under the Target Privacy Budget

1. Use **binary search** to find the privacy parameter **ε** that achieves the target ASR.
2. Run inference under the selected privacy budget:

```bash
bash auto_inferdpt.sh
```
