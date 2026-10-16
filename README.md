# Briefly — AI Article Summarizer

## Deploy to Render
1. Push `main.py`, `requirements.txt`, and optionally `render.yaml` to the root of your GitHub repository.
2. Create a Render Web Service from that repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. In Render → Environment, add `OPENROUTER_API_KEY` with your OpenRouter key.
6. Optional: set `OPENROUTER_MODEL` to `openai/gpt-4o-mini`.

## Run locally
Install dependencies with `pip install -r requirements.txt`, set `OPENROUTER_API_KEY`, then run:
`uvicorn main:app --reload`
Open http://127.0.0.1:8000

Some sites block automated extraction, require JavaScript, or rate-limit requests; no scraper can guarantee every URL will work.
