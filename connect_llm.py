import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# loading the groq llm model
def load_llm():
    llm = ChatGroq(
    groq_api_key = groq_api_key,
    model_name = "llama-3.3-70b-versatile"

    )
    return llm

# Function to create a custom system prompt
def set_custom_prompt(system_message):
    return ChatPromptTemplate.from_messages([
        ("system", system_message),  
        ("user", "Context: {context}"),  
        ("user", "Question: {question}")  
    ])

# Default system prompt
sys_prompt = set_custom_prompt("""Use the piece of information provided in the context to answer the user's question. 
If you don't know the answer or you don't find the answer in the context, just say: 
'The question is not a part of my training data, so I cannot answer.' 
Do not make up an answer.  
Do not provide information outside the context.  
Do not say 'I am not a doctor' or 'consult a doctor'.  
Just answer based on the context. If the answer is available, provide it. Otherwise, follow the instructions above. 
Start your answer directly. No small talk. But if somebody say hi or hello or any such greeting just say
                               Hello there! I AM ADAMAS MEDICAL CHATBOT""")

# Load the database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Creating question andswer chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type='stuff',  
    retriever = db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": sys_prompt}
)

# Now invoke the chain
user_query = input("Write the query here : ")
response = qa_chain.invoke({"query":user_query})

# show the answer
answer = response["result"]
if answer == "Hello there! I AM ADAMAS MEDICAL CHATBOT":
    print("\nHello there! I AM ADAMAS MEDICAL CHATBOT.\n")  
elif answer == "The question is not a part of my training data, so I cannot answer.":
    print("\nThe question is not a part of my training data, so I cannot answer.\n")  
else:
    print(f"\nAnswer: {answer}\n")
    if "source_documents" in response and response["source_documents"]:
        print("Sources:")
        for doc in response["source_documents"]:
            page = doc.metadata.get("page", "N/A") 
            print(f"- Page: {page}")



import pyttsx3

# Initialize the TTS engine
engine = pyttsx3.init()

# Set properties (optional)
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
answer = response.get("answer", "").strip()
# Text to speech
engine.say(answer)
engine.runAndWait()  # Wait for completion

