import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import faiss

# Define paths to your QA pairs and Faiss index
# Please replace these with the actual paths to your files
qa_pairs_path = "output.json"
index_path = "test.index"

e5_model = SentenceTransformer("intfloat/multilingual-e5-large")
with open(qa_pairs_path, 'r', encoding='utf-8') as f:
    bm_mt5_qa_pairs = json.load(f)
bm_mt5_index = faiss.read_index(index_path)


def help_center_rag_inference(query, index, qa_pairs, top_k=10, similarity_threshold=0.8):
    query_embedding = e5_model.encode(["query: " + query])

    # 检索最相似的 10 个记录
    D, I = index.search(np.array(query_embedding), k=top_k)
    # I 返回的是索引位置，D 返回的是距离/相似度分数
    # 将索引和分数与文档一起返回
    retrieved_results = []
    for doc_idx, score in zip(I[0], D[0]):
        if score > similarity_threshold:
            filtered_document = {}
            for key, value in qa_pairs[doc_idx].items():
                if key != "for_embeddings":
                    filtered_document[key] = value
            retrieved_results.append(filtered_document)

    return retrieved_results

# Streamlit app
st.title("Help Center RAG Inference")

query = st.text_input("Enter your query:")

if st.button("Search"):
    if query:
        results = help_center_rag_inference(query, bm_mt5_index, bm_mt5_qa_pairs)
        if results:
            st.subheader("Search Results:")
            for result in results:
                st.write(result) # You might want to format this output more nicely
        else:
            st.write("No results found or similarity scores are below the threshold.")
    else:
        st.write("Please enter a query.")
