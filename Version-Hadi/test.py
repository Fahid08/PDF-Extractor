from hybrid_RAG import model, sentences 
from helpers import generate_dense_embeddings

generate_dense_embeddings(sentences, "How much carbon do we lose each year?", model)