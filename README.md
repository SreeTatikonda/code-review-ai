#  CodeReview AI — Fine-tuned DeepSeek-Coder

A real-time AI code reviewer powered by **DeepSeek-Coder-6.7B** fine-tuned on GitHub PR reviews.  
Streams review feedback token-by-token via WebSockets with a Monaco-style React UI.

---

## 📁 Project Structure

```
code-review-ai/
├── training/
│   ├── build_dataset.py     # Scrapes GitHub PRs → JSONL dataset
│   └── train.py             # QLoRA fine-tuning with TRL + PEFT
├── backend/
│   └── server.py            # FastAPI + WebSocket + vLLM inference
└── frontend/
    └── App.jsx              # React UI with streaming review panel
```

---

##  Step 1: Build Dataset

```bash
cd training/
pip install PyGithub requests tqdm

# Get a GitHub token: https://github.com/settings/tokens
export GITHUB_TOKEN=ghp_your_token_here

python build_dataset.py --output data/code_review_dataset.jsonl --limit 50000
```

Dataset format (JSONL):
```json
{"language": "python", "input": "def login()...", "output": " Critical — SQL Injection..."}
```

Supplement with Microsoft's CodeReviewer dataset:
- Download from: https://github.com/microsoft/CodeBERT/tree/master/CodeReviewer
- Place at: `training/data/microsoft_codereview.jsonl`

---

## Step 2: Fine-tune

**Recommended GPU:** A100 40GB (~$2-3/hr on Vast.ai or RunPod)

```bash
pip install transformers trl peft bitsandbytes datasets accelerate wandb torch

# Set your W&B key (optional but recommended for tracking)
export WANDB_API_KEY=your_key

python train.py --mode train
```

Training config (in `train.py`):
| Parameter | Value |
|-----------|-------|
| Base model | deepseek-ai/deepseek-coder-6.7b-instruct |
| Quantization | 4-bit NF4 (QLoRA) |
| LoRA rank | 16 |
| Batch size | 4 (×4 grad accum = 16 effective) |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Max seq length | 2048 |

Model saved to `./deepseek-coder-review/`

**Test inference after training:**
```bash
python train.py --mode test --model_path ./deepseek-coder-review
```

---

##  Step 3: Run the Backend

```bash
cd backend/
pip install fastapi uvicorn vllm peft transformers torch python-multipart

# With vLLM (recommended for production, requires CUDA)
python server.py --model ../training/deepseek-coder-review --port 8000

# Without vLLM (slower, CPU/MPS compatible)
python server.py --model ../training/deepseek-coder-review --no-vllm
```

API endpoints:
- `GET  /health` — health check
- `POST /review` — non-streaming HTTP review
- `WS   /ws/review` — streaming WebSocket review

---

##  Step 4: Run the Frontend

The frontend is a React app. Create a new Vite project and drop in `App.jsx`:

```bash
npm create vite@latest frontend -- --template react
cd frontend
npm install
# Copy App.jsx → src/App.jsx
npm run dev
```

Or use it as a single-file artifact in Claude.ai.

WebSocket connects to `ws://localhost:8000/ws/review`.

---

## WebSocket Protocol

**Client → Server:**
```json
{
  "code": "def login(username, password):\n  ...",
  "language": "python"
}
```

**Server → Client (streaming):**
```
Review
 tok
en by
 token...
__DONE__
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./deepseek-coder-review` | Path to fine-tuned model |
| `USE_VLLM` | `true` | Use vLLM for inference |
| `GITHUB_TOKEN` | — | For dataset building |
| `WANDB_API_KEY` | — | Weights & Biases tracking |

---

## Roadmap

- [ ] VS Code extension (Language Server Protocol)
- [ ] GitHub PR webhook integration
- [ ] Multi-file context awareness with RAG
- [ ] Severity filtering in UI
- [ ] Export review as markdown/PDF
- [ ] Docker Compose deployment

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Base Model | DeepSeek-Coder-6.7B-Instruct |
| Fine-tuning | QLoRA (PEFT + TRL) |
| Inference | vLLM / HuggingFace Transformers |
| Backend | FastAPI + WebSockets |
| Frontend | React + custom code editor |
| Training tracking | Weights & Biases |
| Dataset | GitHub PR API + MS CodeReviewer |
