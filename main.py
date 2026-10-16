```python
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


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Briefly — AI Article Summarizer</title>

<!-- Markdown renderer and HTML sanitizer -->
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.2.6/dist/purify.min.js"></script>

<style>
* {
    box-sizing: border-box;
}

body {
    margin: 0;
    min-height: 100vh;
    background: #0b1120;
    color: #eef2ff;
    font-family: Arial, sans-serif;
    display: grid;
    place-items: center;
    padding: 24px;
}

main {
    width: 100%;
    max-width: 780px;
}

.brand {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    margin: 0;
}

.brand span {
    color: #818cf8;
}

.sub {
    text-align: center;
    color: #a5b4fc;
    margin: 12px 0 30px;
    line-height: 1.6;
}

form {
    display: flex;
    gap: 10px;
    padding: 8px;
    border: 1px solid #334155;
    background: #172033;
    border-radius: 14px;
}

input {
    flex: 1;
    min-width: 0;
    padding: 14px;
    border: 0;
    outline: 0;
    background: transparent;
    color: white;
    font-size: 15px;
}

button {
    border: 0;
    border-radius: 10px;
    background: #6366f1;
    color: white;
    padding: 14px 18px;
    font-weight: bold;
    cursor: pointer;
}

button:disabled {
    opacity: 0.6;
    cursor: wait;
}

#status {
    min-height: 24px;
    text-align: center;
    color: #fbbf24;
    margin: 18px 0;
    line-height: 1.5;
}

.result {
    display: none;
    margin-top: 22px;
    padding: 24px;
    background: #172033;
    border: 1px solid #334155;
    border-radius: 14px;
    line-height: 1.8;
    overflow-wrap: anywhere;
}

.result h2 {
    margin-top: 0;
    color: #a5b4fc;
}

#summary {
    white-space: normal;
    line-height: 1.8;
}

#summary h1,
#summary h2,
#summary h3 {
    color: #a5b4fc;
    margin-top: 24px;
    margin-bottom: 12px;
    line-height: 1.4;
}

#summary h1 {
    font-size: 26px;
}

#summary h2 {
    font-size: 22px;
}

#summary h3 {
    font-size: 18px;
}

#summary strong {
    color: #ffffff;
}

#summary ul,
#summary ol {
    padding-left: 24px;
}

#summary li {
    margin-bottom: 10px;
}

#summary p {
    margin-bottom: 14px;
}

#summary a {
    color: #a5b4fc;
}

#summary blockquote {
    border-left: 3px solid #6366f1;
    margin-left: 0;
    padding-left: 16px;
    color: #cbd5e1;
}

.note {
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
    margin-top: 24px;
    line-height: 1.6;
}

@media (max-width: 560px) {
    .brand {
        font-size: 40px;
    }

    form {
        flex-direction: column;
    }

    input {
        padding: 14px;
    }

    .result {
        padding: 18px;
    }
}
</style>
</head>

<body>
<main>
    <h1 class="brand">Briefly<span>.</span></h1>

    <p class="sub">
        Paste a news or blog URL and get a clear summary with key takeaways.
    </p>

    <form id="form">
        <input
            type="url"
            name="url"
            placeholder="https://example.com/article"
            required
        >
        <button id="btn" type="submit">Summarize article</button>
    </form>

    <p id="status" role="status" aria-live="polite"></p>

    <section id="result" class="result">
        <h2>Article Summary</h2>
        <div id="summary"></div>
    </section>

    <p class="note">
        Some websites block automated extraction.
        Verify important information with the original article.
    </p>
</main>

<script>
const form = document.getElementById("form");
const btn = document.getElementById("btn");
const status = document.getElementById("status");
const result = document.getElementById("result");
const summary = document.getElementById("summary");

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (typeof marked === "undefined" ||
        typeof DOMPurify === "undefined") {
        status.textContent =
            "Markdown libraries could not load. Please refresh and try again.";
        return;
    }

    btn.disabled = true;
    btn.textContent = "Summarizing...";
    status.textContent =
        "Fetching article and generating summary...";
    result.style.display = "none";
    summary.replaceChildren();

    try {
        const response = await fetch("/summarize", {
            method: "POST",
            body: new FormData(form)
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || "Request failed.");
        }

        // Normalize common escaped Markdown symbols.
        const markdown = String(data.summary || "")
            .replace(/\\\\([*_`#[\\]()])/g, "$1");

        // Convert Markdown into HTML and sanitize it before rendering.
        const renderedHTML = marked.parse(markdown);
        summary.innerHTML = DOMPurify.sanitize(renderedHTML);

        result.style.display = "block";
        status.textContent = "";
    } catch (error) {
        status.textContent =
            error.message || "Something went wrong.";
    } finally {
        btn.disabled = false;
        btn.textContent = "Summarize article";
    }
});
</script>
</body>
</html>
"""


def valid_public_url(url: str) -> bool:
    try:
        parsed = urlparse(url)

        return (
            parsed.scheme in ("http", "https")
            and bool(parsed.hostname)
            and not parsed.username
            and not parsed.password
        )
    except Exception:
        return False


@tool
def fetch_article(url: str) -> str:
    """Fetch and extract readable text from a public news or blog article URL."""

    if not valid_public_url(url):
        return "ERROR: Please provide a valid public HTTP or HTTPS article URL."

    # First attempt: UnstructuredURLLoader
    try:
        loader = UnstructuredURLLoader(
            urls=[url],
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; BrieflySummarizer/1.0)"
            },
            continue_on_failure=False,
        )

        docs = loader.load()

        article_text = "\n\n".join(
            doc.page_content
            for doc in docs
            if doc.page_content
        )

        if len(article_text.strip()) >= 150:
            return article_text[:14000]

    except Exception:
        pass

    # Fallback: requests + BeautifulSoup
    try:
        response = requests.get(
            url,
            timeout=18,
            headers={"User-Agent": "Mozilla/5.0"},
        )

        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()

        if "html" not in content_type:
            return "ERROR: The URL did not return an HTML article page."

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "noscript",
            "svg",
        ]):
            tag.decompose()

        article = soup.find("article") or soup.find("main") or soup.body or soup
        article_text = article.get_text(" ", strip=True)

        if len(article_text) < 150:
            return (
                "ERROR: Not enough article text found. "
                "The site may block automated access or require JavaScript."
            )

        return article_text[:14000]

    except Exception:
        return (
            "ERROR: Could not access this article. "
            "The website may block automated requests or be unavailable."
        )


def make_agent():
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured.")

    llm = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        max_retries=2,
        timeout=75,
    )

    return create_agent(
        model=llm,
        tools=[fetch_article],
        system_prompt=(
            "You are Briefly, an accurate article summarizer. "
            "When given an article URL, call fetch_article first. "
            "If its result begins with ERROR:, report that problem and do not "
            "invent a summary. Otherwise, summarize only the extracted article "
            "text. Start with a short overview, followed by 3-6 key takeaways. "
            "Use valid, standard Markdown formatting. Use ## for section "
            "headings, **text** for bold text, and - for bullet points. "
            "Never escape Markdown symbols with backslashes. "
            "Do not invent facts or include information not supported by "
            "the extracted text. Keep the summary clear and concise."
        ),
    )


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(PAGE)


@app.post("/summarize")
def summarize(url: str = Form(...)):
    url = url.strip()

    if not valid_public_url(url):
        return {
            "error": "Please enter a valid URL starting with http:// or https://."
        }

    if not os.getenv("OPENROUTER_API_KEY"):
        return {
            "error": (
                "Server is missing OPENROUTER_API_KEY. "
                "Add it in Render → Environment."
            )
        }

    try:
        agent = make_agent()

        result = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": f"Summarize this article URL: {url}",
                }
            ]
        })

        answer = ""

        for message in reversed(result.get("messages", [])):
            if (
                getattr(message, "type", "") == "ai"
                and getattr(message, "content", None)
            ):
                answer = message.content

                if isinstance(answer, list):
                    answer = "\n".join(
                        str(item.get("text", item))
                        if isinstance(item, dict)
                        else str(item)
                        for item in answer
                    )

                break

        if not answer:
            return {
                "error": "The AI did not return a summary. Please try again."
            }

        # If the agent returned an extraction error, show it as an error.
        if answer.strip().startswith("ERROR:"):
            return {"error": answer.strip().replace("ERROR:", "", 1).strip()}

        return {"summary": answer}

    except Exception as exc:
        error_text = str(exc).lower()

        if "401" in error_text or "unauthorized" in error_text:
            message = (
                "OpenRouter rejected the API key. "
                "Check OPENROUTER_API_KEY in Render."
            )
        elif (
            "402" in error_text
            or "credit" in error_text
            or "payment required" in error_text
        ):
            message = "OpenRouter credits or billing may be insufficient."
        elif "404" in error_text:
            message = (
                "The configured OpenRouter model may be unavailable. "
                "Check OPENROUTER_MODEL."
            )
        else:
            message = (
                "Summarization failed. Check Render logs for details "
                "and try another article URL."
            )

        return {"error": message}
```
