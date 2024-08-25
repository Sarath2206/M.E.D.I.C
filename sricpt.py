import tensorflow as tf
import numpy as np
import requests
from bs4 import BeautifulSoup
import streamlit as st
from collections import Counter

# Function to scrape data from the Healthy WA website
def scrape_healthywa_data():
    url = 'https://www.healthywa.wa.gov.au/Health-conditions/Health-conditions-A-to-Z'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    conditions = {}
    for condition in soup.select('a[href^="/Health-conditions/Health-conditions-A-to-Z/"]'):
        title = condition.get_text().strip()
        link = condition['href']
        full_url = f'https://www.healthywa.wa.gov.au{link}'

        # Visit each condition page to extract more information
        condition_response = requests.get(full_url)
        condition_soup = BeautifulSoup(condition_response.text, 'html.parser')
        description = condition_soup.find('div', class_='body-content').get_text().strip()

        conditions[title] = description

    return conditions

# Scrape the Healthy WA website data
healthywa_data = scrape_healthywa_data()

# Debugging: Print the first few items
for key in list(healthywa_data.keys())[:5]:
    print(f"Condition: {key}, Description: {healthywa_data[key]}")

# Proceed if data is not empty
if not healthywa_data:
    raise ValueError("Scraped data is empty. Check the scraping process.")

# Tokenize the text and count unique words
tokens = ' '.join(healthywa_data.values()).split()
unique_tokens = Counter(tokens)
vocab_size = len(unique_tokens)

print(f"Tokens: {tokens[:50]}")  # Print first 50 tokens for inspection
print(f"Determined vocabulary size: {vocab_size}")

# Handle case where vocab_size might be 0
if vocab_size == 0:
    raise ValueError("Vocabulary size is 0. Check your tokenization process and ensure that the data is not empty.")

if response.status_code != 200:
    raise ValueError(f"Failed to retrieve data from {url}, status code: {response.status_code}")


# Implement the Transformer Model as defined before
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, head_size, dropout_rate=0.2):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x):
        attn_output = self.attention(x, x)  # Self-attention
        return self.dropout(attn_output)

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, n_embd, dropout_rate=0.2):
        super(FeedForward, self).__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(4 * n_embd, activation='relu'),
            tf.keras.layers.Dense(n_embd),
            tf.keras.layers.Dropout(dropout_rate),
        ])
    
    def call(self, x):
        return self.net(x)

class Block(tf.keras.layers.Layer):
    def __init__(self, n_embd, n_head, dropout_rate=0.2):
        super(Block, self).__init__()
        self.sa = MultiHeadAttention(num_heads=n_head, head_size=n_embd//n_head, dropout_rate=dropout_rate)
        self.ffwd = FeedForward(n_embd, dropout_rate)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()
    
    def call(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

class TransformerModel(tf.keras.Model):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, dropout_rate=0.2):
        super(TransformerModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=n_embd)
        self.blocks = [Block(n_embd, n_head, dropout_rate) for _ in range(n_layer)]
        self.ln_f = tf.keras.layers.LayerNormalization()
        self.head = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def call(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

# Example hyperparameters based on document content
n_embd = 256
n_head = 8
n_layer = 6
dropout_rate = 0.2

model = TransformerModel(vocab_size, n_embd, n_head, n_layer, dropout_rate)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Convert tokens to indices based on the unique_tokens dictionary
token_indices = [unique_tokens[word] for word in tokens]

# Creating sequences for training
sequence_length = 100
x_train = []
y_train = []

for i in range(len(token_indices) - sequence_length):
    x_train.append(token_indices[i:i + sequence_length])
    y_train.append(token_indices[i + 1:i + sequence_length + 1])

x_train = np.array(x_train)
y_train = np.array(y_train)

# Training the model
model.fit(x_train, y_train, epochs=10, batch_size=32)
model.save('transformer_model.h5')

# Streamlit app for RAG Demo
st.title("Retrieval-Augmented Generation (RAG) Demo")

query = st.text_input("Enter your query:")

if query:
    # Retrieve data based on query
    results = [desc for title, desc in healthywa_data.items() if query.lower() in title.lower() or query.lower() in desc.lower()]
    
    if results:
        response = ' '.join(results[:3])  # Display top 3 results
    else:
        response = "No relevant information found in the Healthy WA data."

    # Display the response
    st.write(response)

    # Generate response using the model
    tokenized_query = np.array([[unique_tokens.get(word, 0) for word in query.split()]])  # Tokenization using unique_tokens
    generated_response = model.predict(tokenized_query)

    st.write("Generated Response:")
    st.write(generated_response)
