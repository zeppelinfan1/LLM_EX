import numpy as np


# Sample text
text = "This is a string of text"

# Step 1: Tokenize
tokens = text.lower().split()

# Step 2: Build a vocabulary (The Word is the Key, the Index is the Value)
vocab = {word: i for i, word in enumerate(set(tokens))}

# Step 3: Create an embedding matrix
embedding_dim = 8
embedding_matrix = np.random.randn(len(vocab), embedding_dim)

# Step 4: Convert tokens to embedding vectors
embedded_tokens = np.array([embedding_matrix[vocab[token]] for token in tokens])

for word, index in vocab.items():
    print(f"Word: {word}, Index: {index}, Embedding: {embedding_matrix[index]}")