from LLM_inference import prompt_completions
import os
import joblib
from helpers import generate_sparse_embeddings, generate_dense_embeddings, compute_hybrid_scores

sentence_data_path = "sentence_data.txt"

model = joblib.load('models/sentence_transformer.joblib')

# If sentence file has not been created before
if not os.path.exists(sentence_data_path):
    sentences = []

    # Clean up sentences 
    cleaned_data = prompt_completions.strip('[]')
    prompt_completions_list = [str(x) for x in cleaned_data.split('\n')]

    filtered_list = [item for item in prompt_completions_list if item.startswith('{') and item.endswith('}')]

    for item in filtered_list:
        prompt = item.split('{"prompt":')[1].split(', "completion"')[0]
        completion = item.split('"completion":')[1].split("}")[0]

        sentence = (prompt + completion).replace('"', '')
        sentences.append(sentence)

    # Write sentences to file
    with open(sentence_data_path, 'w', encoding='utf-8') as file:
        for item in sentences:
            file.write(item + '\n')

# If file with clean sentences exists
else:

    # Create embeddings from file
    with open(sentence_data_path, 'r', encoding='utf-8') as file:
        sentences, embeddings = [], []

        for line in file:
            line = line.strip()
            sentences.append(line)


# Generate sparse and dense embeddings
sparse_embeddings = generate_sparse_embeddings(sentences, "How much carbon is lost every year?")
dense_embeddings = generate_dense_embeddings(sentences, "How much carbon is lost every year?", model)

relevant_sparse_samples = {}
relevant_dense_samples = {}

# Return the most relevant sentences indexes according to both sparse and dense similarity thresholds separately
for idx, sentence in sparse_embeddings.items():
    if sentence[1] > 0.3:
        relevant_sparse_samples[idx] = sentence[1]

for idx, sentence in dense_embeddings.items():
    if sentence[1] > 0.5:
        relevant_dense_samples[idx] = sentence[1] 

# Combine indices generated according to both sparse and dense indices
relevant_indices = list(set(relevant_sparse_samples.keys()) | set(relevant_dense_samples.keys()))

# Generate hybrid scores for each relevant index identified
hybrid_scores = compute_hybrid_scores(relevant_indices, sparse_embeddings, dense_embeddings, 0.5)

# Finalized sentences to be used by the RAG
relevant_sentences = [sentences[idx] for idx in hybrid_scores.keys() if hybrid_scores[idx] > 0.4]

print(relevant_sentences)