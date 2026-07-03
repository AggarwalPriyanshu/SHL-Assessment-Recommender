import streamlit as st
import requests
import pandas as pd

BACKEND_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 SHL Assessment Recommendation System")
st.caption("AI-powered SHL Assessment Recommender")

# ------------------------
# Session State
# ------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------
# Sidebar
# ------------------------

with st.sidebar:

    st.header("Conversation")

    if st.button("🔄 Restart Conversation"):

        st.session_state.messages = []
        st.session_state.history = []

        st.rerun()

    st.divider()

    st.write("Backend")

    if st.button("Health Check"):

        try:

            r = requests.get(
                "http://127.0.0.1:8000/health"
            )

            st.success(r.json()["status"])

        except:

            st.error("Backend Offline")
            
            
            
# ------------------------
# Display Chat History
# ------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ------------------------
# User Input
# ------------------------

prompt = st.chat_input(
    "Describe your hiring requirement..."
)

if prompt:

    # ------------------------
    # Show User Message
    # ------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
        "messages": st.session_state.messages
    }

    # ------------------------
    # Call Backend
    # ------------------------

    with st.spinner("Finding the best SHL assessments..."):

        try:

            response = requests.post(
                BACKEND_URL,
                json=payload,
                timeout=60
            )

            data = response.json()
            print(data)

        except Exception as e:  

            st.error(f"Backend Error\n\n{e}")

            st.stop()

    reply = data.get("reply", "No response received.")

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply
    })

    reply = data.get("reply", "No response received.")
    recommendations = data.get("recommendations", [])

    # Save assistant message
    st.session_state.messages.append({
    "role": "assistant",
    "content": reply,
    "recommendations": recommendations
    })

    # Display assistant response
    with st.chat_message("assistant"):

        st.markdown(reply)

        if recommendations:

            st.markdown("### 📋 Recommended Assessments")

            for i, rec in enumerate(recommendations, start=1):

                st.markdown(
                f"""
**{i}. {rec['name']}**

- **Type:** {rec['test_type']}
- **URL:** {rec['url']}

---
"""
            )