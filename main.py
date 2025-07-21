from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional, List
import base64, aiohttp, asyncio, os, json
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PPLX_API_KEY = os.getenv("PPLX_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


async def oai_req(session, b64, prompt, history):
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    messages = []

    # Include previous history for OpenAI
    for entry in history:
        messages.append({"role": "user", "content": entry["prompt"]})
        for res in entry["responses"]:
            if res["provider"] == "openai":
                messages.append({"role": "assistant", "content": res["answer"]})

    # Add new prompt (with optional image)
    content = []
    if prompt:
        content.append({"type": "text", "text": prompt})
    if b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    messages.append({"role": "user", "content": content})

    payload = {"model": "gpt-4o", "messages": messages}

    try:
        async with session.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers) as r:
            res = await r.json()
            if "choices" not in res:
                return {"provider": "openai", "answer": "OpenAI error: missing 'choices'"}
            return {"provider": "openai", "answer": res["choices"][0]["message"]["content"]}
    except Exception:
        return {"provider": "openai", "answer": "OpenAI failed to respond."}


async def pplx_req(session, b64, prompt):
    headers = {"Authorization": f"Bearer {PPLX_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "llava-1.5-13b", "prompt": prompt or ""}
    if b64:
        payload["images"] = [b64]

    try:
        async with session.post("https://api.perplexity.ai/chat/completions", json=payload, headers=headers) as r:
            res = await r.json()
            return {"provider": "perplexity", "answer": res["choices"][0]["message"]["content"]}
    except Exception:
        return {"provider": "perplexity", "answer": "Perplexity failed to respond."}


async def gemini_req(session, b64, prompt):
    headers = {"Content-Type": "application/json"}
    parts = []
    if prompt:
        parts.append({"text": prompt})
    if b64:
        parts.append({"inlineData": {"mimeType": "image/jpeg", "data": b64}})
    payload = {"contents": [{"parts": parts}]}

    try:
        async with session.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
            json=payload,
            headers=headers
        ) as r:
            res = await r.json()
            return {"provider": "gemini", "answer": res["candidates"][0]["content"]["parts"][0]["text"]}
    except Exception:
        return {"provider": "gemini", "answer": "Gemini failed to respond."}


async def fanout(b64img: str, prompt: str, selected_models: List[str], history: List[dict]):
    async with aiohttp.ClientSession() as session:
        tasks = []

        if "openai" in selected_models:
            tasks.append(oai_req(session, b64img, prompt, history))

        if "perplexity" in selected_models:
            tasks.append(pplx_req(session, b64img, prompt))

        if "gemini" in selected_models:
            tasks.append(gemini_req(session, b64img, prompt))

        return await asyncio.gather(*tasks)


@app.post("/vision-query")
async def vision_query(
    prompt: Optional[str] = Form(default=""),
    photo: Optional[UploadFile] = File(default=None),
    models: str = Form(default='["openai", "perplexity", "gemini"]'),
    history: Optional[str] = Form(default="[]")
):
    b64img = ""

    # 🖼️ Read image
    if photo and getattr(photo, "filename", ""):
        try:
            img_bytes = await photo.read()
            b64img = base64.b64encode(img_bytes).decode()
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Failed to read uploaded image."})

    # ❌ Reject empty
    if not prompt.strip() and not b64img:
        return JSONResponse(status_code=400, content={
            "results": [{"provider": "system", "answer": "Please provide at least an image or a prompt."}]
        })

    try:
        selected_models = json.loads(models)
    except Exception:
        selected_models = ["openai", "perplexity", "gemini"]

    try:
        parsed_history = json.loads(history)
    except Exception:
        parsed_history = []

    results = await fanout(b64img, prompt.strip(), selected_models, parsed_history)
    return JSONResponse(content={"results": results})