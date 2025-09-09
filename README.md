# PromptMaskingDemo


# 📘 README — SFT with Prompt Masking

## Why This Script?

This demo shows how to fine-tune a causal language model (like GPT-2, LLaMA, etc.) using **Supervised Fine-Tuning (SFT)** with [TRL’s `SFTTrainer`](https://huggingface.co/docs/trl).

Unlike raw language modeling, instruction-tuning data is structured as:

* **Prompt** → (instruction, input, metadata, etc.)
* **Response** → (desired model output)

If we simply fine-tune on prompt+response concatenated, the model will learn to “predict” the **prompt tokens too**, wasting capacity and hurting alignment.

---

## Why Prompt Masking?

We set the labels for **prompt tokens to `-100`** (ignored by the loss function).

This means:

* The model *sees* the prompt in its input.
* The loss is only computed on the **response tokens**.
* The model learns: “Given this instruction, produce this answer.”

Without masking, the model would waste effort reproducing things like `### Instruction:` or `Count to 3` instead of just learning the correct response.

---

## Why Packing?

Packing combines multiple short examples into a single sequence up to `max_seq_length`.

* **Without packing** → lots of padding, inefficient GPU use.
* **With packing** → dataset fills the context window, faster training, less waste.

---

## Summary

* ✅ **Prompt masking** ensures loss is applied only to responses.
* ✅ **Packing** improves efficiency.
* ✅ **SFTTrainer** simplifies instruction-tuning by handling formatting, batching, and training logic.

This setup is the standard recipe for modern instruction-fine-tuning.

---


# 📝 Demo Script: `train_demo.py`

```python
import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from transformers import DataCollatorForLanguageModeling

# ----------------------------
# Config
# ----------------------------
MODEL = "gpt2"             # small model for demo
TRAIN_FILE = "train.json"   # your dataset (jsonl or json)
OUTPUT_DIR = "./outputs"
MAXSEQ = 128
BATCH_SIZE = 2
EPOCHS = 1
LRATE = 5e-5

# ----------------------------
# Load tokenizer + model
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL)

# ----------------------------
# Dataset
# ----------------------------
# Example train.json content:
# [
#   {"instruction": "Say hello", "output": ["Hello world!"]},
#   {"instruction": "Count to 3", "output": ["1 2 3"]}
# ]

raw_dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")

def formatting_func(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    target = "\n\n".join(example["output"])
    return {"prompt": prompt, "text": prompt + target}

dataset = raw_dataset.map(formatting_func)

# ----------------------------
# Collator with prompt masking
# ----------------------------
class DataCollatorWithPromptMask(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, max_seq_length):
        super().__init__(tokenizer, mlm=False)
        self.max_seq_length = max_seq_length

    def __call__(self, features):
        prompts = [f["prompt"] for f in features]
        texts = [f["text"] for f in features]

        batch = self.tokenizer(texts,
                               truncation=True,
                               padding="max_length",
                               max_length=self.max_seq_length,
                               return_tensors="pt")

        labels = batch["input_ids"].clone()

        # Mask prompt tokens
        for i, prompt in enumerate(prompts):
            prompt_len = len(self.tokenizer(prompt)["input_ids"])
            labels[i, :prompt_len] = -100

        batch["labels"] = labels
        return batch

collator = DataCollatorWithPromptMask(tokenizer, MAXSEQ)

# ----------------------------
# Training args
# ----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LRATE,
    fp16=False,
    bf16=False,
    logging_steps=1,
    save_strategy="no",
    report_to="none"
)

# ----------------------------
# Trainer
# ----------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    max_seq_length=MAXSEQ,
    packing=True,
    data_collator=collator
)

# ----------------------------
# Train
# ----------------------------
trainer.train()

# ----------------------------
# Save final model
# ----------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Training complete. Model saved to", OUTPUT_DIR)
```

---

## 🔧 How to Run

1. Save the script as `train_demo.py`.
2. Create a simple `train.json` file like this:

```json
[
  {"instruction": "Say hello", "output": ["Hello world!"]},
  {"instruction": "Count to 3", "output": ["1 2 3"]}
]
```

3. Run:

```bash
python train_demo.py
```

---

This will:

* Train `gpt2` for 1 epoch on your JSON dataset.
* Apply **prompt masking** so loss is computed only on the response.
* Save model + tokenizer to `./outputs`.

---

👉 Do you want me to also add a **quick evaluation/generation step** at the end (so you can immediately test the fine-tuned model)?
