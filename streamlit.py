import streamlit as st
from langchain.chains import  create_history_aware_retriever, create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from langchain_pinecone import PineconeVectorStore
import re
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
huggingfacehub_api_token = st.secrets["huggingfacehub_api_token"]
pinecone_api_key = st.secrets["pinecone_api_key"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

#model_name = "Alibaba-NLP/gte-multilingual-base"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
#model_kwargs = {'trust_remote_code': True}

# Initialize embeddings
#embedding_mod = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
embedding_mod = HuggingFaceEmbeddings(model_name=model_name)

# Initialize Pinecone
vectorstore = PineconeVectorStore(
    index_name="mental-health-english",
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
    Your primary goal is to promote self-discovery and support the client in exploring their thoughts and feelings without judgment. Always express genuine curiosity and foster a safe space for the client to gradually open up. Ask questions little by little, building rapport while gently encouraging the client to explore their emotions. Always reply in Yoruba.

    Guidelines:

    Focus on empathy: Ask questions that show genuine curiosity, such as:
    "I'm curious about how that made you feel."
    "What thoughts went through your mind when that happened?"
    Avoid leading questions: Instead of implying judgment (e.g., "Don't you think..."), ask neutrally:
    "How do you feel about...?"
    Encourage reflection: Frame questions to help the client explore their own solutions, such as:
    "Can you describe a time when you handled a similar situation more successfully?"
    "How do you see this issue affecting your life?"

    Prompt: Use the following context to answer the client's question. Provide helpful information, and ask one or two reflective questions to guide them further in their journey.

    Context: {context}

    Chat History: {chat_history}

    Client's Question: {input}

    Response: Answer the client's question empathetically, based on the context provided and the chat history. Additionally, ask reflective questions to encourage deeper exploration.
    Always reply in yoruba language.  
    """

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "chat_history", "input"]
)

# Initialize memory
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
st.title("Ẹ káàbọ̀ sí ìtójú ọkàn lórí ìǹtànétì.")

# User info form
if 'user_initialized' not in st.session_state:
    st.session_state['user_initialized'] = False

if not st.session_state['user_initialized']:
    with st.form(key='user_info_form'):
        name = st.text_input("Kí ni orúkọ rẹ?:")
        occupation = st.text_input("Kí ni iṣẹ́ rẹ?:")
        issues = st.text_area("Iru ìṣòrò ọkàn wo ni o n dojú kọ?")
        
        submit_button = st.form_submit_button(label='Bẹrẹ ìjíròrò')

    if submit_button:
        if name and occupation and issues:
            initial_message = f"Orúkọ mi ni {name}, mo ni {occupation}, mo sì ń dojú kọ {issues}."
            response = generate_response(initial_message, st.session_state['session_id'])
            if 'messages' not in st.session_state:
                st.session_state.messages = []
            st.session_state.messages.append({"content": initial_message, "is_user": True})
            st.session_state.messages.append({"content": response, "is_user": False})
            
            st.session_state['user_initialized'] = True
            st.success("Ijíròrò Bẹrẹ!")
        else:
            st.warning("Jọwọ kọ gbogbo awọn aaye ṣaaju ki o to bẹrẹ ìjíròrò.")

# Chat interface after initialization
if st.session_state['user_initialized']:
    # Display chat history
    for message in st.session_state.messages:
        if message["is_user"]:
            st.write(f":bust_in_silhouette: **User:** {message['content']}")
        else:
            st.write(f":robot_face: **Chatbot:** {message['content']}")
    
    # New input box appears under the last message
    user_input = st.text_input("Beere nipa Ilera Ọpọlọ (kọ 'opari' lati parí):", key="user_input")
    if st.button("Submit"):
        if user_input.lower() != 'opari':
            response = generate_response(user_input, st.session_state['session_id'])
            st.session_state.messages.append({"content": user_input, "is_user": True})
            st.session_state.messages.append({"content": response, "is_user": False})
            # Use st.rerun() instead of st.experimental_rerun()
            st.rerun()
        else:
            st.write("Odabo!")
            st.session_state['user_initialized'] = False
            st.session_state.messages = []  # Clear messages when session ends
            st.rerun()

if __name__ == "__main__":
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = "default_session"  # You might want to implement dynamic session IDs
