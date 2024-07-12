from LLM_inference import prompt_completions
import os
import joblib

sentence_data_path = "sentence_data.txt"
embeddings_exist = False

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
    if not embeddings_exist:
        with open(sentence_data_path, 'r', encoding='utf-8') as file:
            sentences, embeddings = [], []

            for line in file:
                line = line.strip()
                sentences.append(line)
                embeddings.append(model.encode(line))
        
        embeddings_exist = True


# client = chromadb.Client()
# collection = client.create_collection(name="sentence_embeddings")

# for i, sentence in enumerate(sentences):
#     collection.add(
#         ids = str(i),
#         documents=[
#             {"id": str(i), "text": sentence, "embedding": embeddings[i].tolist()}
#         ]
#     )

# docs = collection.get()
# for doc in docs:
#     print(f"id: {doc['id']} Text: {doc['text']}, Embedding: {doc['embedding']}")