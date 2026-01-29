# Codes for *Differentially Private and Communication-Efficient Large Language Model Split Inference via Stochastic Quantization and Soft Prompt* (Pangu)

## Environment Setup

Use the official **`pangu`** environment with: **`peft==0.17.1`**


Place all code inside: **`openPangu-Embedded-7B-V1.1`**


Set the environment variable:

```bash
export PANGU_PATH=/path/to/pangu
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/huggingface
````
---

## Download

```python
python download.py
```
---

## Training

### Step 1: Train the Encoder–Decoder Architecture

Run:

```bash
bash train_proj.sh
```

This script trains the encoder–decoder model used for split inference.

---

### Step 2: Search for Privacy Parameter μ and Train the Soft Prompt

1. Perform a **binary search** over the DP mechanism parameter **μ** to reach the target attack success rate (ASR).
2. For the selected **μ**, train the corresponding soft prompt:

```bash
bash train_soft.sh
```

Be sure to replace `--emb_ckpt` with the path where you saved the encoder–decoder weights from **Step 1**.

---

## Testing

### Evaluate Coherence Between Generated Text and Ground Truth

To compute the coherence score between model generations and reference text, run:

```bash
bash test_cse.sh
```

---

## InferDPT Baseline

### Step 1: Export Tokenizer Vocabulary

Enter the `inferdpt` directory:

```bash
cd inferdpt
python export_vocab.py --model openPangu-Embedded-7B-V1.1
```

---

### Step 2: Search for ε and Run Inference Under the Target Privacy Budget

1. Use **binary search** to find the privacy parameter **ε** that achieves the target ASR.
2. Run inference under the selected privacy budget:

```bash
bash auto_inferdpt.sh
```
