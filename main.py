from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional
import base64, aiohttp, asyncio, os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PPLX_API_KEY = os.getenv("PPLX_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---------- External Requests ----------

async def oai_req(session, b64, prompt):
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    content = []
    if prompt:
        content.append({"type": "text", "text": prompt})
    if b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    payload = {"model": "gpt-4o", "messages": [{"role": "user", "content": content}]}
    try:
        async with session.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers) as r:
            res = await r.json()
            return {"provider": "openai", "answer": res.get("choices", [{}])[0].get("message", {}).get("content", "OpenAI gave no response")}
    except:
        return {"provider": "openai", "answer": "OpenAI failed to respond."}


async def pplx_req(session, b64, prompt):
    headers = {"Authorization": f"Bearer {PPLX_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "llava-1.5-13b", "prompt": prompt or ""}
    if b64:
        payload["images"] = [b64]
    try:
        async with session.post("https://api.perplexity.ai/chat/completions", json=payload, headers=headers) as r:
            res = await r.json()
            return {"provider": "perplexity", "answer": res.get("choices", [{}])[0].get("message", {}).get("content", "Perplexity gave no response")}
    except:
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
            return {"provider": "gemini", "answer": res.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Gemini gave no response")}
    except:
        return {"provider": "gemini", "answer": "Gemini failed to respond."}


# ---------- Fanout ----------
async def fanout(b64img: str, prompt: str):
    async with aiohttp.ClientSession() as session:
        return await asyncio.gather(
            oai_req(session, b64img, prompt),
            pplx_req(session, b64img, prompt),
            gemini_req(session, b64img, prompt),
        )


# ---------- Main Endpoint ----------
@app.post("/vision-query")
async def vision_query(
    photo: Optional[UploadFile] = File(default=None),
    prompt: Optional[str] = Form(default="")
):
    b64img = ""

    # Handle image upload (if provided and valid)
    if photo and getattr(photo, "filename", None):
        try:
            img_bytes = await photo.read()
            b64img = base64.b64encode(img_bytes).decode()
        except:
            return JSONResponse(status_code=400, content={"error": "Failed to read uploaded image."})

    # Reject if both are completely empty
    if not prompt.strip() and not b64img:
        return JSONResponse(status_code=400, content={
            "results": [{"provider": "system", "answer": "Please provide at least an image or a prompt."}]
        })

    results = await fanout(b64img, prompt.strip())
    return JSONResponse(content={"results": results})
