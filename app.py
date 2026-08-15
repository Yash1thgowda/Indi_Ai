import os
from typing import TypedDict

from fastapi import FastAPI
from langserve import add_routes

from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END

from pydantic import BaseModel


# ============================================================
# 1. GEMINI API KEY
# ============================================================

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")


# ============================================================
# 2. LLM INITIALIZATION
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key
)


# ============================================================
# 3. INPUT / OUTPUT SCHEMAS
# ============================================================

class IndiAIInput(BaseModel):
    question: str


class IndiAIOutput(BaseModel):
    answer: str


# ============================================================
# 4. LANGGRAPH STATE
# ============================================================

class IndiAIState(TypedDict):
    question: str
    answer: str


# ============================================================
# 5. INDI_AI AGENT NODE
# ============================================================

def indi_ai_node(state: IndiAIState):

    question = state["question"]

    prompt = f"""
You are Indi_Ai, a friendly AI assistant specialized in
Indian history and the history of India's independence.

Your main purpose is to answer questions about:

- Indian history
- Indian freedom struggle
- Indian independence movement
- Indian independence in 1947
- Important freedom fighters
- Mahatma Gandhi
- Subhas Chandra Bose
- Jawaharlal Nehru
- Sardar Vallabhbhai Patel
- Bhagat Singh
- Rani Lakshmibai
- Bal Gangadhar Tilak
- Sarojini Naidu
- Chandrashekhar Azad
- British rule in India
- Major historical events
- Indian Independence Day
- Republic of India
- Important dates and movements

Answer clearly and accurately.

For historical questions:
- Give the important facts.
- Include dates when useful.
- Explain the context in simple language.
- Do not unnecessarily make the answer extremely long.

If the user asks something unrelated to Indian history,
politely explain that Indi_Ai is primarily designed for
Indian history and Independence-related questions.

User's question:

{question}
"""

    response = llm.invoke(prompt)

    # Gemini/LangChain normally returns a string here,
    # but this safely handles structured content too.
    content = response.content

    if isinstance(content, str):
        answer = content

    elif isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))

        answer = "\n".join(parts)

    else:
        answer = str(content)

    return {
        "answer": answer
    }


# ============================================================
# 6. BUILD LANGGRAPH
# ============================================================

workflow = StateGraph(IndiAIState)

workflow.add_node("indi_ai", indi_ai_node)

workflow.add_edge(START, "indi_ai")
workflow.add_edge("indi_ai", END)

indi_ai_graph = workflow.compile()


# ============================================================
# 7. CREATE PUBLIC RUNNABLE
# ============================================================

def run_indi_ai(input_data: IndiAIInput):

    result = indi_ai_graph.invoke({
        "question": input_data.question,
        "answer": ""
    })

    return {
        "answer": result["answer"]
    }


indi_ai_runnable = RunnableLambda(run_indi_ai).with_types(
    input_type=IndiAIInput,
    output_type=IndiAIOutput
)


# ============================================================
# 8. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Indi_Ai",
    version="1.0",
    description="Indi_Ai - Indian History and Independence Q&A Agent"
)


# ============================================================
# 9. LANGSERVE ROUTE
# ============================================================

add_routes(
    app,
    indi_ai_runnable,
    path="/agent"
)


# ============================================================
# 10. BASIC HOME ROUTE
# ============================================================

@app.get("/")
def home():

    return {
        "name": "Indi_Ai",
        "description": "Indian History and Independence Q&A Agent",
        "status": "running",
        "playground": "/agent/playground/"
    }
