import os
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import UnstructuredURLLoader

load_dotenv()
app = FastAPI(title="Briefly AI Article Summarizer")

PAGE = """<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Briefly — AI Article Summarizer</title>
<style>
*{box-sizing:border-box}body{margin:0;min-height:100vh;background:#0b1120;color:#eef2ff;font-family:Arial,sans-serif;display:grid;place-items:center;padding:24px}
main{width:100%;max-width:780px}.brand{text-align:center;font-size:48px;font-weight:800;margin:0}.brand span{color:#818cf8}
.sub{text-align:center;color:#a5b4fc;margin:12px 0 30px;line-height:1.6}
form{display:flex;gap:10px;padding:8px;border:1px solid #334155;background:#172033;border-radius:14px}
input{flex:1;min-width:0;padding:14px;border:0;outline:0;background:transparent;color:white;font-size:15px}
button{border:0;border-radius:10px;background:#6366f1;color:white;padding:14px 18px;font-weight:bold;cursor:pointer}
button:disabled{opacity:.6;cursor:wait}#status{min-height:24px;text-align:center;color:#fbbf24;margin:18px 0}
.result{display:none;margin-top:22px;padding:24px;background:#172033;border:1px solid #334155;border-radius:14px;line-height:1.75;overflow-wrap:anywhere}
.result h2{margin-top:0;color:#a5b4fc}#summary{white-space:pre-wrap}.note{text-align:center;color:#94a3b8;font-size:12px;margin-top:24px}
@media(max-width:560px){.brand{font-size:40px}form{flex-direction:column}input{padding:14px}}
</style></head><body><main>
<h1 class="brand">Briefly<span>.</span></h1><p class="sub">Paste a news or blog URL and get a clear summary with key takeaways.</p>
<form id="form"><input type="url" name="url" placeholder="https://example.com/article" required><button id="btn" type="submit">Summarize article</button></form>
<p id="status" role="status"></p><section id="result" class="result"><h2>Article Summary</h2><div id="summary"></div></section>
<p class="note">Some websites block automated extraction. Verify important information with the original article.</p>
</main><script>
const form=document.getElementById('form'),btn=document.getElementById('btn');
form.addEventListener('submit',async e=>{e.preventDefault();const status=document.getElementById('status'),result=document.getElementById('result'),summary=document.getElementById('summary');
btn.disabled=true;btn.textContent='Summarizing...';status.textContent='Fetching article and generating summary...';result.style.display='none';summary.textContent='';
try{const res=await fetch('/summarize',{method:'POST',body:new FormData(form)});const data=await res.json();if(!res.ok||data.error)throw new Error(data.error||'Request failed.');summary.textContent=data.summary;result.style.display='block';status.textContent='';}
catch(err){status.textContent=err.message||'Something went wrong.'}finally{btn.disabled=false;btn.textContent='Summarize article'}});
</script></body></html>"""

def valid_public_url(url):
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.hostname) and not parsed.username and not parsed.password
    except Exception:
        return False

@tool
def fetch_article(url: str) -> str:
    """Fetch and extract readable text from a public news or blog article URL."""
    if not valid_public_url(url):
        return "ERROR: Please provide a valid public HTTP or HTTPS article URL."
    try:
        loader = UnstructuredURLLoader(
            urls=[url],
            headers={"User-Agent": "Mozilla/5.0 (compatible; BrieflySummarizer/1.0)"},
            continue_on_failure=False,
        )
        docs = loader.load()
        text = "\n\n".join(doc.page_content for doc in docs if doc.page_content)
        if len(text.strip()) >= 150:
            return text[:14000]
    except Exception:
        pass
    # Fallback extraction if UnstructuredURLLoader cannot parse the site.
    try:
        response = requests.get(url, timeout=18, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        if "html" not in response.headers.get("content-type", "").lower():
            return "ERROR: The URL did not return an HTML article page."
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
            tag.decompose()
        article = soup.find("article") or soup.find("main") or soup.body or soup
        text = article.get_text(" ", strip=True)
        if len(text) < 150:
            return "ERROR: Not enough article text found. The site may block automated access or require JavaScript."
        return text[:14000]
    except Exception:
        return "ERROR: Could not access this article. The website may block automated requests or be unavailable."

def make_agent():
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured.")
    llm = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        max_retries=2,
        timeout=75,
    )
    return create_agent(
        model=llm,
        tools=[fetch_article],
        system_prompt=(
            "You are Briefly, an accurate article summarizer. For a URL, call fetch_article first. "
            "If its result begins with ERROR:, report that problem and do not invent a summary. "
            "Otherwise summarize only the extracted text: give a short overview and 3-6 key takeaways. "
            "Do not invent facts."
        ),
    )

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(PAGE)

@app.post("/summarize")
def summarize(url: str = Form(...)):
    url = url.strip()
    if not valid_public_url(url):
        return {"error": "Please enter a valid URL starting with http:// or https://."}
    if not os.getenv("OPENROUTER_API_KEY"):
        return {"error": "Server is missing OPENROUTER_API_KEY. Add it in Render → Environment."}
    try:
        result = make_agent().invoke({"messages": [{"role": "user", "content": f"Summarize this article URL: {url}"}]})
        answer = ""
        for message in reversed(result.get("messages", [])):
            if getattr(message, "type", "") == "ai" and getattr(message, "content", None):
                answer = message.content
                if isinstance(answer, list):
                    answer = "\n".join(str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in answer)
                break
        return {"summary": answer} if answer else {"error": "The AI did not return a summary. Please try again."}
    except Exception as exc:
        err = str(exc).lower()
        if "401" in err or "unauthorized" in err:
            msg = "OpenRouter rejected the API key. Check OPENROUTER_API_KEY in Render."
        elif "402" in err or "credit" in err or "payment required" in err:
            msg = "OpenRouter credits or billing may be insufficient."
        elif "404" in err:
            msg = "The configured OpenRouter model may be unavailable. Check OPENROUTER_MODEL."
        else:
            msg = "Summarization failed. Check Render logs for details and try another article URL."
        return {"error": msg}
