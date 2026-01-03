"""
FastAPI server for the RAG-MCP Assistant.
"""

import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from openai import OpenAI
import requests

app = FastAPI()

# Templates directory for the simple frontend UI
templates = Jinja2Templates(directory="templates")

# API clients / config
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


class QueryRequest(BaseModel):
    query: str
    max_results: int = 5


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Renders templates/index.html
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health_check():
    return {"status": "healthy"}


def web_search_serpapi(query: str, max_results: int = 5) -> str:
    """Call SerpAPI and build a text context from top results."""
    url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": max_results,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("organic_results", [])[:max_results]

    context_lines = []
    for r in results:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        link = r.get("link", "")
        context_lines.append(f"Title: {title}\nSnippet: {snippet}\nURL: {link}")
    return "\n\n".join(context_lines)


@app.post("/query")
async def query(request: QueryRequest):
    # 1. Web search
    context = web_search_serpapi(request.query, request.max_results)

    # 2. Ask OpenAI to synthesize answer
    prompt = (
        "You are an AI assistant. Use ONLY the following web search results to "
        "answer the user question concisely and accurately. At the end, list 2–3 "
        "source URLs you used.\n\n"
        f"Search results:\n{context}\n\n"
        f"User question: {request.query}\n\nAnswer:"
    )

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400,
    )
    answer = completion.choices[0].message.content

    return {
        "query": request.query,
        "max_results": request.max_results,
        "response": answer,
    }

