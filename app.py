import os
from typing import TypedDict

from fastapi import FastAPI
from langserve import add_routes
from pydantic import BaseModel

from langchain_core.runnables import RunnableLambda


from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. GEMINI API KEY
# ============================================================

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not configured")


# ============================================================
# 2. GEMINI LLM
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

For historical questions:
- Give important facts.
- Include dates when useful.
- Explain the context simply.
- Do not make the answer unnecessarily long.

If the question is unrelated to Indian history,
politely explain that Indi_Ai is primarily an Indian
history and Independence-related assistant.

USER QUESTION:

{question}
"""

    response = llm.invoke(prompt)

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


import os
from typing import TypedDict

from fastapi import FastAPI
from langserve import add_routes
from pydantic import BaseModel

from langchain_core.runnables import RunnableLambda


from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. GEMINI API KEY
# ============================================================

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not configured")


# ============================================================
# 2. GEMINI LLM
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

For historical questions:
- Give important facts.
- Include dates when useful.
- Explain the context simply.
- Do not make the answer unnecessarily long.

If the question is unrelated to Indian history,
politely explain that Indi_Ai is primarily an Indian
history and Independence-related assistant.

USER QUESTION:

{question}
"""

    response = llm.invoke(prompt)

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
# 7. FUNCTION THAT CALLS THE GRAPH
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
# 8. CONVERT FUNCTION INTO LANGCHAIN RUNNABLE
# ============================================================

indi_ai_runnable = RunnableLambda(
    ask_indi_ai
).with_types(
    input_type=IndiAIInput,
    output_type=IndiAIOutput
)


# ============================================================
# 9. FASTAPI
# ============================================================

app = FastAPI(
    title="Indi_Ai",
    version="1.0",
    description="Indian History and Independence Q&A Agent"
)


# ============================================================
# 10. LANGSERVE
# ============================================================

add_routes(
    app,
    indi_ai_runnable,
    path="/agent"
)


# ============================================================
# 11. HOME ROUTE
# ============================================================

@app.get("/")
def home():

    return {
        "name": "Indi_Ai",
        "status": "running",
        "message": "Indian History Q&A Agent",
        "playground": "/agent/playground/"
    }

# ============================================================
# 8. CONVERT FUNCTION INTO LANGCHAIN RUNNABLE
# ============================================================

indi_ai_runnable = RunnableLambda(
    ask_indi_ai
).with_types(
    input_type=IndiAIInput,
    output_type=IndiAIOutput
)


# ============================================================
# 9. FASTAPI
# ============================================================

app = FastAPI(
    title="Indi_Ai",
    version="1.0",
    description="Indian History and Independence Q&A Agent"
)


# ============================================================
# 10. LANGSERVE
# ============================================================

add_routes(
    app,
    indi_ai_runnable,
    path="/agent"
)


# ============================================================
# 11. HOME ROUTE
# ============================================================

@app.get("/")
def home():

    return {
        "name": "Indi_Ai",
        "status": "running",
        "message": "Indian History Q&A Agent",
        "playground": "/agent/playground/"
    }
