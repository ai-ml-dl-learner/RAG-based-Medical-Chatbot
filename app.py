import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Load the Groq LLM model
def load_llm():
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile"
    )

# Function to create a custom system prompt
def set_custom_prompt(system_message):
    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "Context: {context}"),
        ("user", "Question: {question}")
    ])

# Load FAISS vector store in cached 
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

def main():
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("adamas.png", width=100)
    with col2:
        st.markdown("<h1 style='text-align: left;'>ADAMAS MEDICAL BOT</h1>", unsafe_allow_html=True)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Enter the Question...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Define system prompt
        sys_prompt = set_custom_prompt("""Use the piece of information provided in the context to answer the user's question. 
        If you don't know the answer or you don't find the answer in the context, just say: 
        'The question is not a part of my training data, so I cannot answer.' 
        Do not make up an answer.  
        Do not provide information outside the context.  
        Do not say 'I am not a doctor' or 'consult a doctor'.  
        Just answer based on the context. If the answer is available, provide it. Otherwise, follow the instructions above. 
        Start your answer directly. No small talk. But if somebody says hi or hello or any such greeting, just say:
        'Hello there! I AM ADAMAS MEDICAL CHATBOT'""")

        try:
            # Load FAISS vector store
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store.")
                return

            # Create retrieval QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type='stuff',
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": sys_prompt}
            )
            result = qa_chain.invoke({"query": prompt})
            answer = result["result"]
            if answer == "Hello there! I AM ADAMAS MEDICAL CHATBOT":
                final_response = "**Hello there! I AM ADAMAS MEDICAL CHATBOT**"
            elif answer == "The question is not a part of my training data, so I cannot answer.":
                final_response = "**The question is not a part of my training data, so I cannot answer.**"
            else:
                pages = list({doc.metadata.get("page", "N/A") for doc in result["source_documents"]})
                source_info = f"\n\n**Sources:** " + ", ".join([f"Page {p}" for p in pages]) if pages else ""
                final_response = f"**Answer:** {answer}{source_info}"
            st.chat_message("assistant").markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
