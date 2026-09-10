import os
import requests
from dotenv import load_dotenv
from dataclasses import dataclass
from tavily import TavilyClient
import json
from datetime import datetime, timedelta

# Loading and verification
load_dotenv()
print("API Key Loaded:", os.getenv("GROQ_API_KEY") is not None)
print("Google API Key Loaded:", os.getenv("GOOGLE_API_KEY") is not None)


# Retrieval tool used by the RAG step below: fetches live results from Tavily
# when the agent's routing decision determines a query needs current information.
class WebSearchTool:
    def __init__(self):
        self.client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    def execute(self, query: str) -> str:

        print("Searching...")
        try:
            response = self.client.search(query=query, search_depth="basic", include_answer=True)

            # return answer or no answer found if not found
            return response.get("answer", "No answer found.")

        except Exception as e:
            print(f"Error during search: {e}")
            return "Failed to search"


# a function calling the LLM API (Groq)
def call_llm(prompt: str):
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:
        print("\nERROR")
        return f"Error calling LLM API: {e}"


# a function calling Google's Gemini API (Google AI Studio key)
def call_gemini(prompt: str, model: str = "gemini-2.5-flash"):
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": os.getenv("GOOGLE_API_KEY"),
            },
            json={
                "contents": [
                    {"parts": [{"text": prompt}]}
                ]
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    except requests.exceptions.RequestException as e:
        print("\nERROR")
        return f"Error calling Gemini API: {e}"
    except (KeyError, IndexError) as e:
        print("\nERROR")
        return f"Unexpected Gemini response format: {e} -- raw: {response.text}"


# Agent Logic: the LLM autonomously decides, per query, whether it needs to search
# the web before answering, then grounds its final answer in the retrieved context (RAG).
def run_agent(user_query: str, llm_fn=call_llm):
    print(f"\nUser Query: '{user_query}'")

    # Autonomous routing step: ask the LLM whether it needs real-time info to answer.
    routing_prompt = f"Do this query require real-time information that you do not have access to? If not, does the following query require a real-time web search to answer? Answer only with 'yes' or 'no'.\n\nQuery: {user_query}"
    decision = llm_fn(routing_prompt).strip().lower()
    print(f"LLM Decision: Search required? -> {decision}")

    # Only search if the agent decided it needs current information.
    if "yes" in decision:
        search_tool = WebSearchTool()
        search_result = search_tool.execute(user_query)

        # RAG: fold the retrieved search context back into a second LLM call
        # so the final answer is grounded in live data instead of the model's own recall.
        final_prompt = (
            "Based on the following context from a web search, please provide a concise answer to the user's query.\n\n"
            f"Context from search:\n"
            "---\n"
            f"{search_result}\n"
            "---\n\n"
            f"User Query: {user_query}"
        )
        final_answer = llm_fn(final_prompt)
    else:
        # If no search is needed, just get a direct answer.
        final_answer = llm_fn(user_query)

    print("\nFinal Answer:")
    print(final_answer)


if __name__ == "__main__":
    # Use Groq (original behavior)
    run_agent("What is the weather in Tokyo?")

    # Use Gemini instead, just by swapping the llm_fn
    run_agent("What is the latest news about AI?", llm_fn=call_gemini)