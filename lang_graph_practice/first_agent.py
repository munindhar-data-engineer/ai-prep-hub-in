# 1. Define a LLM
# load .env file to get the open ai key
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
# Define the Graph State
from langgraph.graph import MessagesState



load_dotenv()

# load the llm

llm = ChatOpenAI(model='gpt-5.4-nano', temperature=0)
print(llm)

# 3. Define Graph Nodes

SYSTEM_PROMPT = """You are an IT support assistant for a Pvt LTD. \
    Your only responsibility is to assist employees with laptop-related issues.\
     Always respond clearly and politely when the user asks about laptop-issues \
     If the user's question is NOT about laptops please respond politely.\
     'Sorry, I can only assist with laptop related issues, please ask questions about your laptop'\
     Do not provide any troubleshooting, suggestions or information."""

def chat_bot(state: MessagesState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {'messages': [response]}

# 4. connect everything into a Graph
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("chatbot", chat_bot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
agent = graph_builder.compile()



if __name__ == "__main__":
    print("Agent ready. Type 'quite' to exit.\n")
    conversation = []
    while True:
        user_input = input("You: ")
        if user_input.lower() in ('quit', 'exit'):
            break
        conversation.append({"role": "user", "content": user_input})
        result = agent.invoke({"messages": conversation})
        conversation = result["messages"]
        print("Agent:", conversation[-1].conent, "\n")