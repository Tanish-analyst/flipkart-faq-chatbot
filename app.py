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


folder_path = "."  # Current directory
index_name = "index"  # because files are: index.faiss and index.pkl

def load_retriever():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        folder_path=".",                   # Files are uploaded in the root directory
        embeddings=embeddings,
        index_name="index",               # Because files are: index.faiss and index.pkl
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



# =============================   define this prompt very very cautiously, due to this it take me 1 hour to get desired results  =============================

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



# =============================     use this if you want to know about what going as context and question to LLM.  =============================



# rag_chain = (
#     {"context": retrieve_docs | format_docs, "question": RunnablePassthrough()}
#     | RunnableLambda(lambda x: (
#         st.code(f"📌 DEBUG CONTEXT:\n\n{x['context']}"),
#         st.code(f"📌 DEBUG QUESTION:\n\n{x['question']}"),
#         x
#     )[-1])  # return original input
#     | prompt
#     | ChatOpenAI(model="gpt-3.5-turbo")
#     | StrOutputParser()
# )



# =============================
# 🎯 Streamlit ChatGPT-like UI
# =============================

st.set_page_config(page_title="Flipkart FAQ Chatbot", page_icon="🤖")
st.title("📦 Flipkart FAQ Chatbot")
st.caption("Ask me anything about Flipkart policies!")


# =============================     if you want to test the retriver, like what chunks it's retrieving based on question you providing in streamlit app  =============================
# =============================     not required in making streamlit app, just for checking purpose  =============================
query = st.text_input("Enter your question to test the retriever:")


if query:
    with st.spinner("Fetching relevant documents..."):
        docs = retriever.get_relevant_documents(query)

    st.subheader("📄 Retrieved Documents:")
    for i, doc in enumerate(docs, 1):
        st.markdown(f"**Document {i}:**")
        st.code(doc.page_content)

# =============================

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
