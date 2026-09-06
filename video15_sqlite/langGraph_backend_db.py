from langgraph.graph import StateGraph ,START,END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,AIMessage,SystemMessage,HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
import sqlite3


load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen3.8-2.4T-A95B",
#     task="text-generation",
#     max_new_tokens=512,
# )

# model = ChatHuggingFace(llm=llm)
model = ChatOllama(
    model="qwen3:4b",
    temperature=0,
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]


def chat_node(state:ChatState):
    messages = state['messages']
    response = model.invoke(messages)

    return {'messages':[response]}

conn = sqlite3.connect(database='chatbot.db',check_same_thread=False)
# checkpointer
checkpointer = SqliteSaver(conn)

## GRAPH
graph=  StateGraph(ChatState)

graph.add_node('chat_node',chat_node)
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)


chatbot = graph.compile(checkpointer = checkpointer)


#test 
# CONFIG = {'configurable': {'thread_id': 'thread-1'}}
# response = chatbot.invoke(
#             {'messages': [HumanMessage(content="What is my name")]},
#             config=CONFIG,
# )
# print(response)

def retrive_all_thread():
    all_thread=set()
    for checkpoint in checkpointer.list(None):
        all_thread.add(checkpoint.config['configurable']['thread_id'])

    return list(all_thread)