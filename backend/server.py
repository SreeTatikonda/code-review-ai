"""
Code Review AI - FastAPI Backend
Serves the fine-tuned DeepSeek-Coder model via WebSocket with streaming output.

Requirements:
    pip install fastapi uvicorn vllm peft transformers torch python-multipart

Run:
    python server.py --model ./deepseek-coder-review --port 8000
"""

import os
import json
import asyncio
import argparse
import logging
from typing import AsyncGenerator
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "./deepseek-coder-review")
USE_VLLM = os.environ.get("USE_VLLM", "true").lower() == "true"

SYSTEM_PROMPT = """You are an expert code reviewer. When given a code snippet, you:
1. Identify bugs, logical errors, and edge cases
2. Flag security vulnerabilities (SQL injection, XSS, CSRF, etc.)
3. Point out performance issues
4. Suggest style and readability improvements
5. Provide corrected code where applicable

Format your review with severity levels: 🚨 Critical | ⚠️ Warning | 💡 Suggestion"""

# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────
class ModelEngine:
    def __init__(self, model_path: str, use_vllm: bool = True):
        self.model_path = model_path
        self.use_vllm = use_vllm
        self.model = None
        self.tokenizer = None
        self.llm = None

    def load(self):
        if self.use_vllm:
            self._load_vllm()
        else:
            self._load_hf()

    def _load_vllm(self):
        """Load with vLLM for production-grade throughput."""
        try:
            from vllm import LLM, SamplingParams
            logger.info(f"Loading model with vLLM: {self.model_path}")
            self.llm = LLM(
                model=self.model_path,
                dtype="bfloat16",
                max_model_len=4096,
                gpu_memory_utilization=0.85,
                trust_remote_code=True,
            )
            self.SamplingParams = SamplingParams
            logger.info("✅ vLLM model loaded")
        except ImportError:
            logger.warning("vLLM not available, falling back to HuggingFace")
            self._load_hf()

    def _load_hf(self):
        """Load with HuggingFace Transformers (fallback, slower)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        logger.info(f"Loading HuggingFace model: {self.model_path}")
        base_model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.eval()
        logger.info("✅ HuggingFace model loaded")

    def build_prompt(self, code: str, language: str) -> str:
        return f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
Please review the following {language} code:

```{language}
{code}
```
<|im_end|>
<|im_start|>assistant
"""

    async def stream_review(self, code: str, language: str) -> AsyncGenerator[str, None]:
        prompt = self.build_prompt(code, language)

        if self.llm:
            yield from self._stream_vllm(prompt)
        else:
            async for token in self._stream_hf(prompt):
                yield token

    def _stream_vllm(self, prompt: str):
        """Synchronous vLLM generation (wrap in thread for async use)."""
        params = self.SamplingParams(
            temperature=0.2,
            top_p=0.95,
            max_tokens=1024,
        )
        outputs = self.llm.generate([prompt], params)
        full_text = outputs[0].outputs[0].text
        # Simulate streaming by yielding word by word
        words = full_text.split(" ")
        for word in words:
            yield word + " "

    async def _stream_hf(self, prompt: str) -> AsyncGenerator[str, None]:
        """Async streaming with HuggingFace generate + streamer."""
        from transformers import TextIteratorStreamer
        import threading

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": 1024,
            "temperature": 0.2,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for token in streamer:
            yield token
            await asyncio.sleep(0)  # yield control to event loop

        thread.join()


# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
engine = ModelEngine(MODEL_PATH, USE_VLLM)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Loading model...")
    engine.load()
    yield
    logger.info("🛑 Shutting down")

app = FastAPI(
    title="Code Review AI",
    description="Real-time AI-powered code review using fine-tuned DeepSeek-Coder",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.websocket("/ws/review")
async def websocket_review(websocket: WebSocket):
    """
    WebSocket endpoint for streaming code review.
    
    Client sends JSON: {"code": "...", "language": "python"}
    Server streams back tokens as plain text, ends with "__DONE__"
    """
    await websocket.accept()
    client = websocket.client.host
    logger.info(f"🔗 WebSocket connected: {client}")

    try:
        while True:
            # Receive code from client
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
                code = data.get("code", "").strip()
                language = data.get("language", "python").strip()
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                continue

            if not code:
                await websocket.send_text(json.dumps({"error": "No code provided"}))
                continue

            logger.info(f"📝 Reviewing {language} code ({len(code)} chars) from {client}")

            # Stream the review token by token
            try:
                async for token in engine.stream_review(code, language):
                    await websocket.send_text(token)
                await websocket.send_text("__DONE__")
            except Exception as e:
                logger.error(f"Generation error: {e}")
                await websocket.send_text(f"\n\n❌ Error during generation: {str(e)}")
                await websocket.send_text("__DONE__")

    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected: {client}")


# ─────────────────────────────────────────────
# HTTP fallback (non-streaming)
# ─────────────────────────────────────────────
class ReviewRequest(BaseModel):
    code: str
    language: str = "python"

class ReviewResponse(BaseModel):
    review: str
    language: str

@app.post("/review", response_model=ReviewResponse)
async def review_code_http(request: ReviewRequest):
    """Non-streaming HTTP endpoint (for testing)."""
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="No code provided")

    full_review = ""
    async for token in engine.stream_review(request.code, request.language):
        full_review += token

    return ReviewResponse(review=full_review.strip(), language=request.language)


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--no-vllm", action="store_true")
    args = parser.parse_args()

    engine.model_path = args.model
    engine.use_vllm = not args.no_vllm

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
