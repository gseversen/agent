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


# This class handles my web search logic, using a search API called Tavily 
class WebSearchTool:
    def __init__(self):
        self.client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    def execute(self, query: str) -> str:

        print("Searching...")
        try:
            response = self.client.search(query=query, search_depth="basic",include_answer=True)
            
            # return answer or no answer found if not found
            return response.get("answer", "No answer found.")

        except Exception as e:
            print(f"Error during search: {e}")
            return "Failed to search"


# a function calling the LLM API
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


# Agent Logic 
def run_agent(user_query: str):
    print(f"\nUser Query: '{user_query}'")

    #LLM decides if it requires a tool 
    routing_prompt = f"Do this query require real-time information that you do not have access to? If not, does the following query require a real-time web search to answer? Answer only with 'yes' or 'no'.\n\nQuery: {user_query}"
    decision = call_llm(routing_prompt).strip().lower()
    print(f"LLM Decision: Search required? -> {decision}")

    # search the web if need be.
    if "yes" in decision:
        search_tool = WebSearchTool()
        search_result = search_tool.execute(user_query)

        # funnel the web result back into the LLM for an answer (RAG)
        final_prompt = (
            "Based on the following context from a web search, please provide a concise answer to the user's query.\n\n"
            f"Context from search:\n"
            "---\n"
            f"{search_result}\n"
            "---\n\n"
            f"User Query: {user_query}"
        )
        final_answer = call_llm(final_prompt)
    else:
        # If no search is needed, just get a direct answer.
        final_answer = call_llm(user_query)

    print("\nFinal Answer:")
    print(final_answer)



if __name__ == "__main__":
    run_agent("WWhat is the weather in Tokyo?")
    run_agent("what is the quickest way to add a new user to a Linux system?")
    run_agent("What is the best way to learn Python?")