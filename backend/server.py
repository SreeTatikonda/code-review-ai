import json, asyncio, argparse, logging, httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-coder:6.7b"
MOCK_MODE = False

MOCK_REVIEW = """🚨 Critical — SQL Injection vulnerability on line 3. User input is directly interpolated into the query string.

Fix:
```python
query = "SELECT * FROM users WHERE username=? AND password=?"
result = db.execute(query, (username, password))
```

⚠️ Warning — Never compare passwords in plain text. Use bcrypt:
```python
import bcrypt
if bcrypt.checkpw(password.encode(), user.hashed_password):
    ...
```

💡 Suggestion — Add input validation for empty username/password before querying the database."""

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Mock mode ON" if MOCK_MODE else f"Using Ollama: {OLLAMA_MODEL}")
    yield

app = FastAPI(title="Code Review AI", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

async def stream_ollama(code, language):
    prompt = f"Review this {language} code and identify bugs, security issues, and improvements. Use 🚨 Critical, ⚠️ Warning, 💡 Suggestion prefixes.\n\n```{language}\n{code}\n```"
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True, "options": {"temperature": 0.2}}) as r:
            async for line in r.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if data.get("response"):
                            yield data["response"]
                        if data.get("done"):
                            break
                    except: continue

async def stream_mock(code, language):
    for word in MOCK_REVIEW.split(" "):
        yield word + " "
        await asyncio.sleep(0.04)

@app.get("/health")
async def health():
    return {"status": "ok", "mode": "mock" if MOCK_MODE else "ollama"}

@app.websocket("/ws/review")
async def ws_review(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = json.loads(await websocket.receive_text())
            code = data.get("code", "").strip()
            language = data.get("language", "python")
            if not code:
                continue
            try:
                streamer = stream_mock(code, language) if MOCK_MODE else stream_ollama(code, language)
                async for token in streamer:
                    await websocket.send_text(token)
            except Exception as e:
                logger.error(e)
                async for token in stream_mock(code, language):
                    await websocket.send_text(token)
            await websocket.send_text("__DONE__")
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    MOCK_MODE = args.mock
    uvicorn.run(app, host="0.0.0.0", port=args.port)
