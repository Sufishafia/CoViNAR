import pandas as pd
df  = pd.read_csv("tweet_file.csv")

#RoBERTa Embeddings
!pip install transformers
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
import torch
import numpy as np

# Load the RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

# Create an empty list to store the embeddings
embeddings = []

# Assuming you have a DataFrame 'df' with a column 'ProcessedText' containing your tweets
for tweet in df['ProcessedText']:
    # Tokenize the tweet and add special tokens
    input_ids = tokenizer.encode(tweet, add_special_tokens=True)

    # Convert input to a PyTorch tensor
    input_ids = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension

    # Forward pass through the model to get embeddings
    with torch.no_grad():
      outputs = model(input_ids)

    # Extract the embedding for the [CLS] token (the first token)
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    # Convert the PyTorch tensor to a numpy array
    cls_embedding = cls_embedding.numpy()

    # Append the embedding to the list
    embeddings.append(cls_embedding)

# Convert the list of embeddings to a NumPy array
X = np.array(embeddings)

# Now, the 'embeddings' array contains the embeddings for each tweet



#DistilBERT Embeddings

!pip install transformers
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np

# Load the DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Create an empty list to store the embeddings
embeddings = []

# Loop through the tweets and extract embeddings
for tweet in df['ProcessedText']:
    # Tokenize the tweet and add special tokens
    input_ids = tokenizer.encode(tweet, add_special_tokens=True)

    # Convert input to a PyTorch tensor
    input_ids = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension

    # Forward pass through the model to get embeddings
    with torch.no_grad():
        outputs = model(input_ids)

    # Extract the embedding for the [CLS] token (the first token)
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    # Convert the PyTorch tensor to a numpy array
    cls_embedding = cls_embedding.numpy()

    # Append the embedding to the list
    embeddings.append(cls_embedding)

# Convert the list of embeddings to a NumPy array
X = np.array(embeddings)

# Now, the 'embeddings' array contains the embeddings for each tweet
# print(embeddings)



#BERT-Base embeddings

# Install the Transformers library if you haven't already
!pip install transformers

import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Create an empty list to store the embeddings
embeddings = []

# Assuming you have a DataFrame 'df' with a column 'ProcessedText'
for tweet in df['ProcessedText']:
    # Tokenize the tweet and add special tokens
    input_ids = tokenizer.encode(tweet, add_special_tokens=True)

    # Convert input to a PyTorch tensor
    input_ids = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension

    # Forward pass through the model to get embeddings
    with torch.no_grad():
        outputs = model(input_ids)

    # Extract the embedding for the [CLS] token (the first token)
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    # Convert the PyTorch tensor to a numpy array
    cls_embedding = cls_embedding.numpy()

    # Append the embedding to the list
    embeddings.append(cls_embedding)

# Convert the list of embeddings to a NumPy array
X = np.array(embeddings)

#X contains the embeddings of tweets


