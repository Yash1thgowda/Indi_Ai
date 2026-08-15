import os
from typing import TypedDict

from fastapi import FastAPI
from langserve import add_routes
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. API KEY
# ============================================================

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not configured")


# ============================================================
# 2. GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key
)


# ============================================================
# 3. INPUT / OUTPUT
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
# 5. INDI_AI NODE
# ============================================================

def indi_ai_node(state: IndiAIState):

    question = state["question"]

    prompt = f"""
You are Indi_Ai, an AI assistant specializing in Indian history.

You are especially designed for India's Independence Day
and the Indian freedom struggle.

Answer the user's question clearly, accurately, and in a
student-friendly way.

You can answer questions about:

- Indian history
- British rule in India
- Indian freedom struggle
- Indian independence
- Independence Day
- Mahatma Gandhi
- Subhas Chandra Bose
- Bhagat Singh
- Jawaharlal Nehru
- Sardar Vallabhbhai Patel
- Rani Lakshmibai
- Sarojini Naidu
- Bal Gangadhar Tilak
- Chandrashekhar Azad
- Indian independence movements
- Important historical events and dates

If appropriate, include important dates and context.

If the question is unrelated to Indian history, politely explain
that you are primarily an Indian history assistant.

USER QUESTION:

{question}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# ============================================================
# 6. BUILD LANGGRAPH
# ============================================================

workflow = StateGraph(IndiAIState)

workflow.add_node(
    "indi_ai",
    indi_ai_node
)

workflow.add_edge(
    START,
    "indi_ai"
)

workflow.add_edge(
    "indi_ai",
    END
)

indi_ai_graph = workflow.compile()


# ============================================================
# 7. PUBLIC FUNCTION
# ============================================================

def ask_indi_ai(input_data: IndiAIInput):

    result = indi_ai_graph.invoke(
        {
            "question": input_data.question,
            "answer": ""
        }
    )

    return IndiAIOutput(
        answer=result["answer"]
    )


# ============================================================
# 8. FASTAPI
# ============================================================

app = FastAPI(
    title="Indi_Ai",
    version="1.0",
    description="Indian History Q&A Agent"
)


# ============================================================
# 9. LANGSERVE
# ============================================================

add_routes(
    app,
    ask_indi_ai,
    path="/agent"
)


# ============================================================
# 10. HOME
# ============================================================

@app.get("/")
def home():

    return {
        "name": "Indi_Ai",
        "status": "running",
        "message": "Indian History Q&A Agent",
        "playground": "/agent/playground/"
    }
