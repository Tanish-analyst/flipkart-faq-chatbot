# 🤖 Flipkart FAQ Chatbot (RAG-Based with LangChain + OpenAI)

An intelligent customer support chatbot that answers **Flipkart FAQs** using **RAG (Retrieval-Augmented Generation)**.  
Built with **LangChain**, **OpenAI**, and deployed on **Streamlit**, this bot leverages real FAQs from Flipkart’s Help Center to provide precise and relevant responses to common customer queries.

---

## 📌 What Can This Chatbot Answer?

The chatbot is trained to handle questions from the following Flipkart Help Center categories:

- 🚚 **Delivery**
- 💳 **Payments**
- 👤 **Login & Account**
- 🪙 **SuperCoins**
- 💸 **Refunds**
- ❌ **Cancellations**

For example, you can ask:

- "How can I track my delivery?"
- "What happens if a payment fails?"
- "How do I reset my Flipkart password?"
- "How are SuperCoins credited to my account?"
- "How long does a refund take?"
- "How to cancel a product after placing the order?"

---

## 🧠 Tech Stack

- **🦜 LangChain** – RAG pipeline, prompt chaining
- **🔮 OpenAI** – Embeddings & Chat model (`gpt-3.5-turbo`)
- **📦 FAISS** – Vector store for fast semantic document retrieval
- **📄 Flipkart Help Center** – Real FAQ data source
- **🌐 Streamlit** – Chatbot UI for interactive experience

---

## 🚀 How It Works

1. **User Query** ➝ User asks a question via chat.
2. **Retriever** ➝ FAISS retrieves the most relevant FAQ chunks based on the query.
3. **Prompting** ➝ LangChain feeds context + question into a prompt template.
4. **LLM Response** ➝ OpenAI generates a clear, helpful answer.
5. **Streamlit Chat UI** ➝ Response is displayed in a chatbot-like interface.

---
