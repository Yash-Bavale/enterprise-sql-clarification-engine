from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from pydantic import BaseModel

# 1. Define the State
class State(TypedDict):
    question: str
    is_ambiguous: bool
    clarification_message: str
    sql_query: str

class AmbiguityCheck(BaseModel):
    is_ambiguous: bool
    clarification_message: str

llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

# 2. Define the Nodes
def check_ambiguity(state: State):
    prompt = f"Analyze this request for a sales DB: '{state['question']}'. Is it ambiguous? (e.g., 'best' could mean revenue or volume). If yes, ask a clarification question. If no, leave message blank."
    structured_llm = llm.with_structured_output(AmbiguityCheck)
    res = structured_llm.invoke(prompt)
    return {"is_ambiguous": res.is_ambiguous, "clarification_message": res.clarification_message}

def ask_human(state: State):
    # Placeholder node. LangGraph will pause execution BEFORE this node runs.
    pass

def generate_sql(state: State):
    prompt = f"Write Postgres SQL for: {state['question']}. Table schema: sales(id, client_name, revenue, units_sold, date). Return ONLY the raw SQL string."
    res = llm.invoke(prompt)
    return {"sql_query": res.content}

# 3. Define the Routing Logic
def router(state: State):
    if state["is_ambiguous"]:
        return "ask_human"
    return "generate_sql"

# 4. Build and Compile the Graph
workflow = StateGraph(State)
workflow.add_node("check_ambiguity", check_ambiguity)
workflow.add_node("ask_human", ask_human)
workflow.add_node("generate_sql", generate_sql)

workflow.set_entry_point("check_ambiguity")
workflow.add_conditional_edges("check_ambiguity", router)
workflow.add_edge("ask_human", "generate_sql")
workflow.add_edge("generate_sql", END)

# The checkpointer allows the graph to pause and remember state
memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["ask_human"])