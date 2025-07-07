import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import os

# 🔐 Securely load your OpenAI API key  
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


def load_retriever():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        folder_path="index",                 
        embeddings=embeddings,
        index_name="index",            
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": 2})

# ✅ Load the retriever
retriever = load_retriever()

# 📄 Format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 💬 RAG Chain
retrieve_docs = RunnableLambda(retriever.invoke)




prompt = ChatPromptTemplate.from_template(
    """You are a Flipkart customer support assistant.
Use the context below to answer the customer's question **completely and accurately**.
Always include **all important steps or instructions** from the context.
If the answer is not found in the context, say: "Sorry, I couldn't find that information."

{context}

Question: {question}
"""
)

rag_chain = (
    {"context": retrieve_docs | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-3.5-turbo")
    | StrOutputParser()
)




# =============================
# 🎯 Streamlit ChatGPT-like UI
# =============================

st.set_page_config(page_title="Flipkart FAQ Chatbot", page_icon="🤖")
st.title("📦 Flipkart FAQ Chatbot")
st.caption("Ask me anything about Flipkart policies!")



# 🧠 Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
if prompt := st.chat_input("Ask about Flipkart policies..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(prompt)
            st.markdown(response)

    # Save bot message
    st.session_state.messages.append({"role": "assistant", "content": response})
