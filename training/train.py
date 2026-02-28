"""
Fine-tuning DeepSeek-Coder-6.7B-Instruct for Code Review
Using QLoRA (4-bit quantization + LoRA adapters)

Requirements:
    pip install transformers trl peft bitsandbytes datasets accelerate wandb torch
    
GPU: Tested on A100 40GB (Vast.ai / RunPod ~$2-3/hr)
Training time: ~8-12 hours for 50k samples, 3 epochs
"""

import os
import torch
import wandb
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_ID = "deepseek-ai/deepseek-coder-6.7b-instruct"
OUTPUT_DIR = "./deepseek-coder-review"
DATASET_PATH = "./data/code_review_dataset.jsonl"   # your dataset
RUN_NAME = "deepseek-coder-review-v1"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

BATCH_SIZE = 4
GRAD_ACCUM = 4       # effective batch = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_SEQ_LEN = 2048
WARMUP_RATIO = 0.05

# ─────────────────────────────────────────────
# PROMPT FORMAT (DeepSeek-Coder instruct format)
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert code reviewer. When given a code snippet, you:
1. Identify bugs, logical errors, and edge cases
2. Flag security vulnerabilities (SQL injection, XSS, CSRF, etc.)
3. Point out performance issues
4. Suggest style and readability improvements
5. Provide corrected code where applicable

Format your review with severity levels: 🚨 Critical | ⚠️ Warning | 💡 Suggestion"""

def format_prompt(sample):
    """Format dataset sample into DeepSeek-Coder instruct format."""
    language = sample.get("language", "")
    code = sample["input"]
    review = sample["output"]

    prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
Please review the following {language} code:

```{language}
{code}
```
<|im_end|>
<|im_start|>assistant
{review}<|im_end|>"""
    return {"text": prompt}


# ─────────────────────────────────────────────
# LOAD & PREPARE DATASET
# ─────────────────────────────────────────────
def load_and_prepare_data(path: str):
    """
    Expects JSONL with fields: input, output, language (optional)
    
    Example record:
    {
        "language": "python",
        "input": "def get_user(id):\n  query = f'SELECT * FROM users WHERE id={id}'\n  return db.execute(query)",
        "output": "🚨 Critical - SQL Injection: ..."
    }
    
    Data sources to build this dataset:
    - Microsoft CodeReviewer: https://github.com/microsoft/CodeBERT/tree/master/CodeReviewer
    - GitHub API: scrape PRs from popular repos
    - CodeSearchNet: https://github.com/github/CodeSearchNet
    """
    dataset = load_dataset("json", data_files=path, split="train")
    dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    print(f"Train: {len(dataset['train'])} | Eval: {len(dataset['test'])}")
    return dataset


# ─────────────────────────────────────────────
# MODEL SETUP (QLoRA)
# ─────────────────────────────────────────────
def load_model_and_tokenizer(model_id: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def apply_lora(model):
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train():
    wandb.init(project="deepseek-code-review", name=RUN_NAME)

    dataset = load_and_prepare_data(DATASET_PATH)
    model, tokenizer = load_model_and_tokenizer(MODEL_ID)
    model = apply_lora(model)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        learning_rate=LEARNING_RATE,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        logging_steps=25,
        report_to="wandb",
        run_name=RUN_NAME,
        group_by_length=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
    )

    print("🚀 Starting training...")
    trainer.train()

    print("💾 Saving model...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Model saved to {OUTPUT_DIR}")
    wandb.finish()


# ─────────────────────────────────────────────
# INFERENCE TEST (post-training)
# ─────────────────────────────────────────────
def test_inference(model_path: str, code_snippet: str, language: str = "python"):
    from peft import PeftModel

    print("Loading fine-tuned model for inference test...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
Please review the following {language} code:

```{language}
{code_snippet}
```
<|im_end|>
<|im_start|>assistant
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("\n=== Code Review ===")
    print(response)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--model_path", default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        test_code = """
def login(username, password):
    user = db.query(f"SELECT * FROM users WHERE username='{username}'")
    if user and user.password == password:
        session['user'] = username
        return redirect('/dashboard')
    return render_template('login.html', error='Invalid credentials')
"""
        test_inference(args.model_path, test_code, "python")
