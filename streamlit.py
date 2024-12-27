import streamlit as st
from langchain.chains import ConversationalRetrievalChain, create_history_aware_retriever, create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import PromptTemplate
import getpass
import google.generativeai as genai
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import re
import os

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

if 'GOOGLE_API_KEY' not in os.environ:
    os.environ['GOOGLE_API_KEY'] = st.text_input('Provide your Google API Key:', type='password')

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Contextualize question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state:
        st.session_state[session_id] = ChatMessageHistory()
    return st.session_state[session_id]

# Configuration
huggingfacehub_api_token = os.getenv('huggingfacehub_api_token')
pinecone_api_key = os.getenv('pinecone_api_key_2')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

model_name = "Alibaba-NLP/gte-multilingual-base"
model_kwargs = {'trust_remote_code': True}

# Initialize embeddings
embedding_mod = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# Initialize Pinecone
vectorstore = PineconeVectorStore(
    index_name="mental-health-yoruba",
    embedding=embedding_mod,
    pinecone_api_key=pinecone_api_key
)

# Define the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", convert_system_message_to_human=True, temperature=0.3)

# Define the retriever for history-aware retrieval
retriever = vectorstore.as_retriever()
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Define the prompt template
prompt_template = """You are a compassionate and empathetic Yoruba counselor/therapist chatbot. 
... (rest of the prompt template)"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "chat_history", "input"]
)

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=5)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Initialize the ConversationalRetrievalChain
qa = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

def generate_response(user_input, session_id):
    conversational_qa = RunnableWithMessageHistory(
        qa,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    response = conversational_qa.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})
    cleaned_response = re.sub(r"^\s*[-–—]+\s*", "", response['answer'])
    cleaned_response = cleaned_response.replace("\n", " ")
    return cleaned_response.strip()

# Streamlit UI with initial form
st.title("Welcome to Therapeutic Chatbot")

# User info form
if 'user_initialized' not in st.session_state:
    st.session_state['user_initialized'] = False

if not st.session_state['user_initialized']:
    with st.form(key='user_info_form'):
        name = st.text_input("Your Name:")
        occupation = st.text_input("Your Occupation:")
        issues = st.text_area("What mental health issues are you facing?")
        
        submit_button = st.form_submit_button(label='Start Chat')

    if submit_button:
        if name and occupation and issues:
            initial_message = f"My name is {name}, I am a {occupation}, and I'm dealing with {issues}."
            response = generate_response(initial_message, st.session_state['session_id'])
            st.write(f"Chatbot: {response}")
            
            st.session_state['user_initialized'] = True
            st.success("Chat started!")
        else:
            st.warning("Please fill out all fields before starting the chat.")

# Chat interface after initialization
if st.session_state['user_initialized']:
    user_input = st.text_input("Ask about Mental Health (type 'exit' to end):")
    if st.button("Submit"):
        if user_input.lower() != 'exit':
            response = generate_response(user_input, st.session_state['session_id'])
            st.write(f"Chatbot: {response}")
        else:
            st.write("Goodbye!")
            st.session_state['user_initialized'] = False

if __name__ == "__main__":
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = "default_session"  # You might want to implement dynamic session IDs