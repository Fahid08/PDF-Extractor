import camelot
import pandas as pd
import fitz
import re
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

## Text
def is_table_row(line):
    line_text = "".join(span["text"] for span in line["spans"])
    
    pattern = r'(\d+)|([-\s;,.]+)|([A-Za-z]\s[A-Za-z])'
    # r'^[0-9!@#$%^&*()\-_=+\[\]{};:\'",.<>/?\\|]+$'
    
    # case to handle lines containing only numbers and special characters
    if re.search(pattern, line_text) and len(line_text.split()) <= 10 :
        return True
    # case to handle text lines that are not a part of unstructured text
    if len(line_text.split()) < 5 and not line_text.strip().endswith('.'):
        return True
    
    return False


def extract_text(data_file_path):
    doc = fitz.open(data_file_path)
    text_data = ""

    # adjust number of pages to extract
    for page_num in range(5):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block["type"] == 0:  
                for line in block.get("lines", []):
                    if not is_table_row(line):
                        line_text = "".join(span["text"] for span in line["spans"])
                        text_data += line_text + "\n"

    return text_data

## Tables
def extract_tables_from_pdf(data_file_path, flavor='lattice'):    
    tables = camelot.read_pdf(data_file_path, pages='all', flavor=flavor, strip_text='\n')

    table_csv_path = 'extracted_tables.csv'
    
    dfs = []

    with open(table_csv_path, 'w', encoding='utf-8') as f:

        for i, table in enumerate(tables):
            df = table.df
            if stream_is_table(df):
                df.columns = df.iloc[0]
                df = df[1:].reset_index(drop=True)
                dfs.append(df)

                df.to_csv(f, index=False)

                if i < len(tables) - 1:
                    f.write('\n\n')

    return dfs

def stream_is_table(df):
    if df.shape[0] < 2 or df.shape[1] < 2:
        return False

    delimiter_counts = df.apply(lambda x: x.str.count(r'\s+').mean(), axis=1)
    if delimiter_counts.var() > 1:
        return False

    contains_numbers = df.apply(lambda x: x.str.contains('\d').any(), axis=1)
    contains_text = df.apply(lambda x: x.str.contains('[A-Za-z]').any(), axis=1)
    if not (contains_numbers.any() and contains_text.any()):
        return False

    return True

def generate_sparse_embeddings(sentences, user_query):
    sparse_embeddings = {}
    # sparse_embeddings_path = "embeddings/sparse_embeddings.npy"

    # Initialize vectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)

    # Generate an embedding of the user query and find the similarity values for each sentence
    query_embedding = vectorizer.transform([user_query])
    similarities = cosine_similarity(X, query_embedding)

    # Rank sentences by their similarity score, return sentences along with their sparse embedding scores
    ranked_indices = np.argsort(similarities, axis=0)[::-1].flatten()

    for idx in ranked_indices:
        sparse_embeddings[idx] = (sentences[idx], similarities[idx])

    return sparse_embeddings

def generate_dense_embeddings(sentences, user_query, embedding_model):
    dense_embeddings = {}
    dense_embeddings_path = "embeddings/dense_embeddings.npy"
                        
    # Only generate embeddings if they have not already been saved
    if os.path.exists(dense_embeddings_path):
        sentence_embeddings = np.load(dense_embeddings_path)
    else:
        sentence_embeddings = np.array(embedding_model.encode(sentences))
        
        with open(dense_embeddings_path, 'w', encoding='utf-8') as f:
            np.save(dense_embeddings_path, sentence_embeddings)
    
    query_embedding = np.array(embedding_model.encode(user_query))

    # Reshape 1D query embedding to 2D (384,) -> (384, 1)
    y = query_embedding.shape
    query_embedding = query_embedding.reshape(1, int(y[0]))

    similarities = cosine_similarity(sentence_embeddings, query_embedding)

    ranked_indices = np.argsort(similarities, axis=0)[::-1].flatten()

    # Rank sentences by their similarity score, return sentences along with their sparse embedding scores
    for idx in ranked_indices:
        dense_embeddings[idx] = (sentences[idx], similarities[idx])

    return dense_embeddings

def compute_hybrid_scores(relevant_indices, sparse_embeddings, dense_embeddings, alpha):
    hybrid_score_dict = {}

    # Computes a hybrid score to return the most relevant sentences when compared with the user query
    for idx in relevant_indices:
        sparse_score = sparse_embeddings[idx][1]
        dense_score = dense_embeddings[idx][1]

        hybrid_score = (1-alpha)*sparse_score + alpha*(dense_score)

        hybrid_score_dict[idx] = hybrid_score

    return hybrid_score_dict

    
