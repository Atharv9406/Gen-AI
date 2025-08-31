import streamlit as st
import requests
import torch
from sentence_transformers import SentenceTransformer, util
import re

PERPLEXITY_API_KEY = "Your_key"

device = torch.device("cpu")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

notes = {
    "What is Python?": "Python is a high-level, interpreted programming language used for general-purpose programming.",
    "What is AI?": "Artificial Intelligence (AI) is the simulation of human intelligence in machines.",
    "What is Machine Learning?": "Machine Learning is a subset of AI that enables systems to learn and improve from experience without explicit programming."
}

note_texts = list(notes.values())
embeddings = model.encode(note_texts, convert_to_tensor=True, device=device)


def fetch_from_perplexity(query, context="", style="detailed"):
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    system_prompt = (
        "You are a helpful tutor. Provide clear, natural explanations without citations or references. "
        "Be concise and to the point. Never add options or over-explain unless the user specifically asks for more information." if style == "concise"
        else "You are a helpful tutor. Provide clear, detailed explanations without citations or references."
    )

    data = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Use this context (if relevant): {context}"},
            {"role": "user", "content": query}
        ],
        "max_tokens": 300,
        "temperature": 0.3
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            raw_answer = result["choices"][0]["message"]["content"]

            # Remove citation markers like [1], [2], [1][3], etc.
            clean_answer = re.sub(r"\[\d+(?:,\d+)*\]", "", raw_answer)
            clean_answer = re.sub(r"\[\d+\]", "", clean_answer)

            return clean_answer.strip()
        else:
            return "⚠️ Unexpected API response format."
    else:
        return f"⚠️ Perplexity API error: {response.text}"


def hybrid_answer(query, threshold=0.6, style="detailed"):
    query_emb = model.encode(query, convert_to_tensor=True, device=device)
    cos_scores = util.cos_sim(query_emb, embeddings)[0]

    best_idx = torch.argmax(cos_scores).item()
    best_score = cos_scores[best_idx].item()

    if best_score >= threshold:
        return f"📘 From Notes:\n\n{note_texts[best_idx]}"
    else:
        context = note_texts[best_idx]  
        return f"The answer is:\n\n{fetch_from_perplexity(query, context, style)}"

st.title("🤖 Student Doubt Solving Chatbot")
st.write("Ask any question. If it's in your notes, I'll answer from them. Otherwise, I'll fetch from Perplexity AI.")

# Style toggle
style = st.radio("Answer Style", ["Detailed", "Concise"], horizontal=True)

query = st.text_input("💬 Enter your question:")

if query:
    answer = hybrid_answer(query, style=style.lower())
    st.markdown(answer)
