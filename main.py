from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import base64, asyncio, os, aiohttp
from dotenv import load_dotenv
from typing import Optional, Union

load_dotenv()
app = FastAPI()

# Load API keys from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PPLX_API_KEY = os.getenv("PPLX_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


async def oai_req(session, b64, prompt):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    content = [{"type": "text", "text": prompt}]
    if b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": content}]
    }

    try:
        async with session.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers) as r:
            res = await r.json()
            if 'choices' not in res:
                print("OpenAI raw response:", res)
                return {"provider": "openai", "answer": "OpenAI error: missing 'choices'"}
            return {"provider": "openai", "answer": res['choices'][0]['message']['content']}
    except Exception as e:
        print("OpenAI exception:", e)
        return {"provider": "openai", "answer": "OpenAI failed to respond."}


async def pplx_req(session, b64, prompt):
    headers = {
        "Authorization": f"Bearer {PPLX_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"model": "llava-1.5-13b", "prompt": prompt}
    if b64:
        payload["images"] = [b64]

    try:
        async with session.post("https://api.perplexity.ai/chat/completions", json=payload, headers=headers) as r:
            if r.status != 200:
                text = await r.text()
                print("Perplexity API error:", text)
                return {"provider": "perplexity", "answer": f"Perplexity error: {r.status}"}
            res = await r.json()
            return {"provider": "perplexity", "answer": res['choices'][0]['message']['content']}
    except Exception as e:
        print("Perplexity exception:", e)
        return {"provider": "perplexity", "answer": "Perplexity failed to respond."}


async def gemini_req(session, b64, prompt):
    headers = {
        "Content-Type": "application/json"
    }

    parts = [{"text": prompt}]
    if b64:
        parts.append({
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": b64
            }
        })

    payload = {
        "contents": [{"parts": parts}]
    }

    try:
        async with session.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
            json=payload,
            headers=headers
        ) as r:
            if r.status != 200:
                text = await r.text()
                print("Gemini API error:", text)
                return {"provider": "gemini", "answer": f"Gemini error: {r.status}"}
            res = await r.json()
            return {"provider": "gemini", "answer": res['candidates'][0]['content']['parts'][0]['text']}
    except Exception as e:
        print("Gemini exception:", e)
        return {"provider": "gemini", "answer": "Gemini failed to respond."}


async def fanout(b64img: str, prompt: str):
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            oai_req(session, b64img, prompt),
            pplx_req(session, b64img, prompt),
            gemini_req(session, b64img, prompt)
        )
    return results


@app.post("/vision-query")
async def vision_query(
    prompt: str = Form(...),
    photo: Optional[UploadFile] = File(None)
):
    if not prompt.strip():
        return JSONResponse(status_code=400, content={
            "results": [{"provider": "system", "answer": "Prompt is required."}]
        })

    b64img = ""
    if photo:
        try:
            img_bytes = await photo.read()
            b64img = base64.b64encode(img_bytes).decode()
        except Exception as e:
            print("Image read error:", e)
            return JSONResponse(status_code=400, content={"error": "Failed to read uploaded image."})

    results = await fanout(b64img, prompt)
    return JSONResponse(content={"results": results})
