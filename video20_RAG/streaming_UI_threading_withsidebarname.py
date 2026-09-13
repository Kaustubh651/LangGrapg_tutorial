import streamlit as st
from typing import Any

from langraph_rag_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig

## we will dynamically generate thread so we use library called uuid 
import uuid

# --> ***************************** utility function **************************
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id= generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id,"New Chat")
    st.session_state['message_history'] = []

def add_thread(thread_id,thread_name):
    if not any(
        thread['thread_id'] == thread_id
        for thread in st.session_state['chat_threads']
        ):
        st.session_state['chat_threads'].append({
            'thread_id':thread_id,
            'thread_name':thread_name
        })

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    if state is None or not state.values:
        return []
    return state.values.get('messages', [])

# ************************* Session setup  *****************************

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []


# --> adding thread id in session setup

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = [
        {
            "thread_id": thread_id,
            "thread_name": "Previous Chat"
        }
        for thread_id in retrieve_all_threads()
    ]

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state['thread_id'],'New Chat')

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None


# *************************** Side bar UI *******************************


st.sidebar.title("LangGraph PDF Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

st.sidebar.subheader("Past conversations")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for thread_id in threads:
        if st.sidebar.button(str(thread_id), key=f"side-thread-{thread_id}"):
            selected_thread = thread_id


st.sidebar.header("My Convsersations")


for thread in st.session_state['chat_threads'][::-1]:
    thread_id = thread['thread_id']
    thread_name = thread['thread_name']

    if st.sidebar.button(thread_name,key=thread_id):
        st.session_state['thread_id'] =thread_id
        messages = load_conversation(thread_id)

        temp_message= []
        for message in messages:
            if isinstance(message,HumanMessage):
                role='user'
            else:
                role = 'assistant'
            temp_message.append({'role':role,'content':message.content})

        st.session_state['message_history']=temp_message


# *************************** MAIN UI ********************************
st.title("Multi Utility Chatbot")
# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

#{'role': 'user', 'content': 'Hi'}
#{'role': 'assistant', 'content': 'Hi=ello'}

user_input = st.chat_input("Ask about your document or use tools")

if user_input:

    if len(st.session_state['message_history'])==0:
        for thread in st.session_state['chat_threads']:
            if thread['thread_id'] == st.session_state['thread_id']:
                thread['thread_name']= user_input
                break

    

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG: RunnableConfig = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder: dict[str, Any] = {"box": None}
        assistant_content: list[str] = []

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage):
                    content = message_chunk.content
                    if isinstance(content, str):
                        assistant_content.append(content)
                        yield content
                    elif isinstance(content, list):
                        text = "".join(
                            str(part) for part in content if isinstance(part, str)
                        )
                        if text:
                            assistant_content.append(text)
                            yield text

        st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

        st.session_state["message_history"].append(
            {"role": "assistant", "content": "".join(assistant_content)}
        )

        doc_meta = thread_document_metadata(thread_key)
        if doc_meta:
            st.caption(
                f"Document indexed: {doc_meta.get('filename')} "
                f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
            )

    st.divider()

    if selected_thread:
        st.session_state["thread_id"] = selected_thread
        messages = load_conversation(selected_thread)

        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp_messages
        st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
        st.rerun()