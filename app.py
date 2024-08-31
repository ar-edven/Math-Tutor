import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
from agents import Agent, get_videos  # Assuming your Agent class code is in a file named agent_code.py

# Load environment variables
load_dotenv()

# Initialize Streamlit session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Set up Streamlit page
st.set_page_config(page_title="LangGraph Bot", page_icon="🤖")
st.title("LangGraph Bot")

# Instantiate the agent with your model and tools
model = ChatOpenAI(model="gpt-4o-mini")  # Use your preferred model
tool = get_videos  # Use your existing YouTube tool
template = """
    you are a helpful assistant. Keep that in mind use Latex format for maths 
    and also add '$$' at the beginning and end of your LaTeX response 
    
    for example:
    $$ \boxed{\left( x, y \right) = \left( \frac{1}{5}, -\frac{3}{5} \right)} $$ 
    now use this as a sample example and do it throughout the response.
    """
abot = Agent(model, [tool], system=template)

# Function to get a response from the LangGraph-based agent
def get_response(query, chat_history):
    messages = chat_history + [HumanMessage(content=query)]
    result = abot.graph.invoke({"messages": messages})
    return result['messages'][-1].content

# Display conversation history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)

# User input section
user_query = st.chat_input("Your Message")

if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    # Display user message
    with st.chat_message("Human"):
        st.markdown(user_query)

    # Get AI response and display it
    with st.chat_message("AI"):
        ai_response = get_response(user_query, st.session_state.chat_history)
        r = fr'''{ai_response}'''
        st.markdown(r)
        st.session_state.chat_history.append(AIMessage(content=ai_response))
