# agent

A personal AI assistant that gives an open-source LLM autonomous access to real-time web
search through a retrieval-augmented generation (RAG) pipeline.

The agent isn't just "an LLM that calls a search API." For every query, it first decides
for itself — via a routing prompt to the LLM — whether it actually needs current
information it doesn't already have. Only when it decides the answer requires fresh data
does it call out to [Tavily](https://tavily.com) to retrieve live search results. Those
results are then folded back into a second LLM call as grounding context, and the model
uses them to produce the final answer (the RAG step). If the routing decision comes back
"no," the query goes straight to the LLM with no search involved.

Inference runs on [Groq](https://groq.com)'s hosted Llama 3.1 (`llama-3.1-8b-instant`)
for sub-second response times. A Gemini backend is also included and can be swapped in by
passing a different `llm_fn`.

## How it works

1. **Routing decision** — the user query is sent to the LLM with a yes/no prompt asking
   whether answering it requires real-time information the model doesn't have.
2. **Conditional search** — only if the LLM answers "yes," `WebSearchTool` queries Tavily
   and gets back a synthesized answer from the search results.
3. **Grounded generation (RAG)** — the search result is embedded into a second prompt as
   context, and the LLM generates a final answer grounded in that live data. If no search
   was needed, the original query is answered directly.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file with your API keys:

```
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
GOOGLE_API_KEY=your_google_key   # optional, only needed for the Gemini backend
```

## Usage

```bash
python agent.py
```

`run_agent(query, llm_fn=call_llm)` accepts an `llm_fn` so the underlying model can be
swapped — pass `call_gemini` to run the same routing + RAG flow on Gemini instead of Groq.
