
import streamlit as st
import json
import faiss
import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    DPRContextEncoder, DPRContextEncoderTokenizer,
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
    T5ForConditionalGeneration, T5Tokenizer
)

@st.cache_data
def load_corpus():
    with open("/content/corpus.json", "r") as file:
        return json.load(file)

@st.cache_resource
def load_models():
    context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

    question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

    t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")

    return context_encoder, context_tokenizer, question_encoder, question_tokenizer, t5_model, t5_tokenizer

@st.cache_resource
def build_faiss_index(_corpus, _context_encoder, _context_tokenizer):
    encoded_corpus = []
    texts = [doc.get("body", "") for doc in _corpus if doc.get("body")]

    for i in tqdm(range(0, len(texts), 16)):  # Batch processing
        batch_texts = texts[i:i + 16]
        inputs = _context_tokenizer(batch_texts, return_tensors="pt", max_length=256, padding=True, truncation=True)
        with torch.no_grad():
            outputs = _context_encoder(**inputs)
        encoded_corpus.append(outputs.pooler_output.detach().numpy())

    if encoded_corpus:
        encoded_corpus = np.vstack(encoded_corpus)
        index = faiss.IndexFlatIP(768)
        faiss.normalize_L2(encoded_corpus)  # Normalize the vectors
        index.add(encoded_corpus)
        return index
    else:
        return None

def retrieve_documents(query, question_encoder, question_tokenizer, index, corpus):
    inputs = question_tokenizer(query, return_tensors="pt", max_length=512, truncation=True)
    question_embedding = question_encoder(**inputs).pooler_output.detach().cpu().numpy()
    D, I = index.search(question_embedding, k=3)
    return [corpus[i] for i in I[0]]

def generate_answer(query, retrieved_docs, t5_model, t5_tokenizer):
    context = " ".join([doc.get("body", "") for doc in retrieved_docs])
    input_text = f"question: {query} context: {context}"
    inputs = t5_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = t5_model.generate(inputs["input_ids"], max_length=150)
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

def format_output(query, answer, evidence_docs):
    evidence_list = [
        {
            "title": doc["title"],
            "author": doc["author"],
            "url": doc.get("url", "N/A"),
            "source": doc.get("source", "N/A"),
            "category": doc.get("category", "N/A"),
            "published_at": doc.get("published_at", "N/A"),
            "fact": doc["body"][:200]
        }
        for doc in evidence_docs
    ]
    return {
        "query": query,
        "answer": answer,
        "question_type": "inference_query",
        "evidence_list": evidence_list
    }

def main():
    st.title("Document Retrieval and QA System")
    st.write("This app retrieves relevant documents from a corpus and generates an answer to your query using a T5-based model.")

    corpus = load_corpus()
    context_encoder, context_tokenizer, question_encoder, question_tokenizer, t5_model, t5_tokenizer = load_models()
    index = build_faiss_index(corpus, context_encoder, context_tokenizer)

    query = st.text_input("Enter your query:")
    if st.button("Retrieve Documents and Generate Answer"):
        if query:
            retrieved_docs = retrieve_documents(query, question_encoder, question_tokenizer, index, corpus)
            answer = generate_answer(query, retrieved_docs, t5_model, t5_tokenizer)
            output = format_output(query, answer, retrieved_docs)
            st.write("### Answer:")
            st.write(answer)
            st.write("### Supporting Documents:")
            for doc in output["evidence_list"]:
                st.write(f"**Title**: {doc['title']}")
                st.write(f"**Author**: {doc['author']}")
                st.write(f"**Source**: {doc['source']}")
                st.write(f"**Published At**: {doc['published_at']}")
                st.write(f"**Fact**: {doc['fact']}")
                st.write("---")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
