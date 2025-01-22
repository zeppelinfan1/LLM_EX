import numpy as np, pandas as pd
pd.set_option("display.width", 1000)
pd.set_option("display.max_columns", 100)
import os


"""PEPARING DF DATA
"""

# Text from Wine Kaggle DF
df = pd.read_csv(r"C:\Users\DanielC\Desktop\winemag-data-130k-v2.csv", index_col=0)
# Retain only relevent columns
df = df[["description", "points"]]
df["description"] = df["description"].str.lower()


"""CREATING EMBEDDING
"""

# Step 1: Tokenize
full_tokens = []
df["description"].apply(lambda x: [full_tokens.append(word) for word in x.split()])
full_tokens.append("[EOS]")
full_tokens.append("[PAD]")

# Step 2: Build a vocabulary (The Word is the Key, the Index is the Value)
vocab = {word: i for i, word in enumerate(set(full_tokens))}


# Step 3: Create an embedding matrix
embedding_dim = 8
embedding_matrix = np.random.randn(len(vocab), embedding_dim) # 2 extra: 1 for EOS, 1 for PAD

# for word, index in vocab.items():
#     print(f"Word: {word}, Index: {index}, Embedding: {embedding_matrix[index]}")


"""APPLYING EMBEDDING TO DF
"""

# Step 4: Convert tokens to embedding vectors
def apply_embedding(text, seq_length=10):

    tokens = text.split() + ["[EOS]"]
    tokens = tokens[:seq_length]
    tokens += ["[PAD]"] * (seq_length - len(tokens))
    embeddings = [embedding_matrix[vocab[token]].tolist() for token in tokens]

    return embeddings

df["embedding"] = df["description"].apply(apply_embedding)

print(df["embedding"][0])