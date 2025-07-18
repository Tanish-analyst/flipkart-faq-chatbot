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

# =============================
# 📁 Load the retriever
# =============================
def load_retriever():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        folder_path="index",                 
        embeddings=embeddings,
        index_name="index",            
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": 2})

retriever = load_retriever()

# =============================
# 🧠 Define the RAG Chain
# =============================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retrieve_docs = RunnableLambda(retriever.invoke)

chat_prompt = ChatPromptTemplate.from_template(
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
    | chat_prompt
    | ChatOpenAI(model="gpt-3.5-turbo")
    | StrOutputParser()
)

# =============================
# 🎨 Streamlit App UI
# =============================

st.set_page_config(page_title="Flipkart FAQ Chatbot", page_icon="🤖")

# 🎯 Custom header
st.markdown("""
    <div style="text-align: center; padding: 10px;">
        <h1 style="color:#0072E3;">📦 Flipkart FAQ Chatbot</h1>
        <p style="font-size: 18px;">Ask about <b>Deliveries, Payments, Cancellations, Refunds, SuperCoins, Accounts</b></p>
        <hr style="border: 1px solid #f0f0f0;"/>
    </div>
""", unsafe_allow_html=True)

# 📘 Sidebar Help
with st.sidebar:
    st.title("📚 Help Topics")
    st.markdown("""
    - 🚚 Delivery  
    - 💳 Payments  
    - ❌ Cancellations  
    - 💸 Refunds  
    - 🪙 SuperCoins  
    - 👤 Login & Account  
    """)
    st.markdown("---")
    st.markdown("🧠 Powered by LangChain + OpenAI")

# 🧠 Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# 🧾 Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 💬 Chat input
if user_input := st.chat_input("Ask about Flipkart policies..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(user_input)
            st.markdown(response)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
