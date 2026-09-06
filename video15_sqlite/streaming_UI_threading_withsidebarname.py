import streamlit as st
from langGraph_backend_db import chatbot,retrive_all_thread
from langchain_core.messages import HumanMessage

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

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrive_all_thread()

add_thread(st.session_state['thread_id'],'New Chat')



# *************************** Side bar UI *******************************


st.sidebar.title("LangGraph Chatbot")


if st.sidebar.button("New Chat"):
    reset_chat()



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
# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

#{'role': 'user', 'content': 'Hi'}
#{'role': 'assistant', 'content': 'Hi=ello'}

user_input = st.chat_input('Type here')

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

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}
    assistant_reply = []

    def stream_assistant_response():
        for message_chunk, _ in chatbot.stream(
            {'messages': [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode='messages'
        ):
            content = getattr(message_chunk, 'content', '')
            if isinstance(content, str):
                assistant_reply.append(content)
                yield content

    with st.chat_message('assistant'):
        ai_message = st.write_stream(stream_assistant_response())

    st.session_state['message_history'].append({
        'role': 'assistant',
        'content': ''.join(assistant_reply) if assistant_reply else str(ai_message or '')
    })
