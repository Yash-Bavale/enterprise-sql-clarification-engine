import uuid
import streamlit as st
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from pydantic import BaseModel

st.set_page_config(page_title="Text-to-SQL Clarification Engine", layout="centered")
st.title("Enterprise SQL Clarification Engine")

# 1. Define LLM & Pydantic Model
llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0)

class AmbiguityCheck(BaseModel):
    is_ambiguous: bool
    clarification_message: str

# 2. Define State
class State(TypedDict):
    question: str
    is_ambiguous: bool
    clarification_message: str
    sql_query: str

# 3. Define Nodes
def check_ambiguity(state: State):
    question_text = state.get("question", "")
    prompt = f"Analyze this request for a sales DB: '{question_text}'. Is it ambiguous? (e.g., 'best' could mean revenue or volume). If yes, ask a clarification question. If no, leave message blank."
    structured_llm = llm.with_structured_output(AmbiguityCheck)
    res = structured_llm.invoke(prompt)
    return {"is_ambiguous": res.is_ambiguous, "clarification_message": res.clarification_message}

def ask_human(state: State):
    pass

def generate_sql(state: State):
    question_text = state.get("question", "")
    prompt = f"Write Postgres SQL for: {question_text}. Table schema: sales(id, client_name, revenue, units_sold, date). Return ONLY the raw SQL string without markdown formatting."
    res = llm.invoke(prompt)
    content = res.content.strip()
    if content.startswith("```sql"):
        content = content[6:]
    if content.endswith("```"):
        content = content[:-3]
    return {"sql_query": content.strip()}

# 4. Routing Logic
def router(state: State):
    if state.get("is_ambiguous", False):
        return "ask_human"
    return "generate_sql"

# 5. Build and Compile Graph
@st.cache_resource
def get_graph():
    workflow = StateGraph(State)
    workflow.add_node("check_ambiguity", check_ambiguity)
    workflow.add_node("ask_human", ask_human)
    workflow.add_node("generate_sql", generate_sql)

    workflow.set_entry_point("check_ambiguity")
    workflow.add_conditional_edges("check_ambiguity", router)
    workflow.add_edge("ask_human", "generate_sql")
    workflow.add_edge("generate_sql", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory, interrupt_before=["ask_human"])

app = get_graph()

# 6. Session & Navigation Logic
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

def reset_session():
    st.session_state["thread_id"] = str(uuid.uuid4())

with st.sidebar:
    st.title("Navigation")
    if st.button("🏠 Home / New Query", use_container_width=True):
        reset_session()
        st.rerun()

config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

user_input = st.chat_input("Ask a question (e.g., 'Who is our best client?')")

if user_input:
    app.invoke({"question": user_input, "is_ambiguous": False}, config)

# Retrieve graph state safely
state = app.get_state(config)

if state and state.next == ('ask_human',):
    st.warning("⚠️ Ambiguity Detected in Your Request")
    st.write(state.values.get("clarification_message", ""))
    
    clarification = st.text_input("Provide clarification:")
    if st.button("Submit Clarification"):
        current_q = state.values.get("question", "")
        new_q = f"{current_q} (Context: {clarification})"
        app.update_state(config, {"question": new_q, "is_ambiguous": False})
        app.invoke(None, config)
        st.rerun()

elif state and state.values.get("sql_query"):
    st.success("✅ Query Generated")
    sql = state.values["sql_query"]
    st.code(sql, language="sql")
    
    try:
        conn = st.connection("neon", type="sql")
        df = conn.query(sql, ttl=0)
        
        st.write("### Query Results")
        st.dataframe(df)
        
    except Exception as e:
        st.error(f"Database Error: {e}")

    st.write("---")
    if st.button("🔍 Ask Another Question"):
        reset_session()
        st.rerun()
