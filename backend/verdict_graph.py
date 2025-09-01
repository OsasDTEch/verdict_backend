from typing import Annotated, List
from typing_extensions import TypedDict
from backend.agents.search_agent import legal_research_agent
from backend.agents.synthesizer import final_agent
from langgraph.checkpoint.memory import InMemorySaver
import uuid
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from sqlalchemy.orm import Session
from backend.database.models import Conversation

# --- State definition ---
class VerdictState(TypedDict):
    user_input: str
    search_result: str
    synthesizer_output: str
    messages: Annotated[list, add_messages]

# --- Memory and config ---
memory = InMemorySaver()
unique_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": unique_id}}

# --- Graph builder ---
verdictai_graphbuilder = StateGraph(VerdictState)


async def search_info(state: VerdictState):
    # Await the agent run to get the actual result
    search_out = await legal_research_agent.run(state['user_input'])

    # Convert to string or extract 'answer' if it's a dict
    if isinstance(search_out, dict):
        state['search_result'] = search_out.get("answer", "")
    else:
        state['search_result'] = str(search_out)

    return state


async def synthesizer(state: VerdictState):
    synthesized = await final_agent.run(state['search_result'])

    # Direct access since we know the structure
    try:
        state['synthesizer_output'] = synthesized.output.summary
    except AttributeError:
        # Fallback if structure is different
        state['synthesizer_output'] = str(synthesized)

    return state


# --- Build graph edges ---
verdictai_graphbuilder.add_node('search_info', search_info)
verdictai_graphbuilder.add_node('synthesizer', synthesizer)
verdictai_graphbuilder.add_edge(START, "search_info")
verdictai_graphbuilder.add_edge('search_info', "synthesizer")
verdictai_graphbuilder.add_edge('synthesizer', END)

verdictai_graph = verdictai_graphbuilder.compile(checkpointer=memory)

# --- Helper: last messages ---
def get_last_messages(db: Session, user_id: int, limit: int = 20):
    msgs = (
        db.query(Conversation)
        .filter(Conversation.user_id == user_id)
        .order_by(Conversation.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [{"role": m.role, "content": m.message} for m in reversed(msgs)]

# --- Helper: prepare state ---
def prepare_state(user_input: str, messages: List[dict]):
    return {
        "user_input": user_input,
        "messages": messages + [{"role": "user", "content": user_input}],
        "search_result": "",
        "synthesizer_output": ""
    }

# --- Main: run graph ---
import asyncio


async def run_verdict_graph_async(user_input: str, db: Session, user_id: int):
    last_msgs = get_last_messages(db, user_id)
    state = prepare_state(user_input, last_msgs)
    outputs = []

    async for event in verdictai_graph.astream(
            state, config={"configurable": {"thread_id": str(uuid.uuid4())}}
    ):
        for value in event.values():
            if "synthesizer_output" in value and value["synthesizer_output"]:
                outputs.append(value["synthesizer_output"])

    # --- Save messages ---
    db.add(Conversation(user_id=user_id, role="user", message=user_input))
    for msg in outputs:
        db.add(Conversation(user_id=user_id, role="assistant", message=msg))
    db.commit()

    return outputs[-1] if outputs else ""


# --- Optional: streaming output to console ---
def stream_graph_updates(user_input: str):
    for event in verdictai_graph.stream(
        {"user_input": user_input, "messages": [{"role": "user", "content": user_input}]},
        config=config
    ):
        for value in event.values():
            if "synthesizer_output" in value and value["synthesizer_output"]:
                print("\n📌 Synthesized Answer:\n", value["synthesizer_output"])
            elif "messages" in value and value["messages"]:
                print("Assistant (intermediate):", value["messages"][-1].content)
