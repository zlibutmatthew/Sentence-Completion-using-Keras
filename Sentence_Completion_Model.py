# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/zlibutmatthew/Sentence-Completion-using-Keras/blob/main/Sentence_Completion_Model.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + id="JQYtBuPRPxf9"
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
import tensorflow as tf
from tensorflow import keras
import torch

# + [markdown] id="07IvQxIDR5cK"
# ## Importing the Data

# + colab={"base_uri": "https://localhost:8080/"} id="5h2ylKDnQgtM" outputId="b4042f9a-875f-4098-80ab-f2926bcdefdd"
train_df1 = pd.read_csv('train.csv')
train_ls1=train_df1['text'].tolist()
train_ls1[0:5]

# + colab={"base_uri": "https://localhost:8080/"} id="Gf-iYptwRKUW" outputId="07dc1153-1a09-46a0-d2ff-5fc4866f8925"
test_df1 = pd.read_csv('test.csv')
test_ls1=test_df1['Text'].tolist()
print(len(test_ls1))

# + [markdown] id="rj9-N43hR8NM"
# ## Preprocessing

# + colab={"base_uri": "https://localhost:8080/", "height": 35} id="zNm6Dl2RR0ek" outputId="e1591618-88b5-48ef-9906-6d614530f6fb"
#unknown character at index 20
test_ls1.pop(20)

# + colab={"base_uri": "https://localhost:8080/"} id="HNz593x7SCZI" outputId="1dabf2db-fa8d-45e8-c423-df93df73504f"
train_st=''
for item in train_ls1:
    train_st += ' ' + item + '.'

print(len(train_st))

# + colab={"base_uri": "https://localhost:8080/"} id="g4iASTxTSHTD" outputId="a98a17ed-694c-436e-a6b3-752eb5860be1"
# Join all the sentences together and extract the unique characters from the combined sentences
chars = set(train_st)

# Creating a dictionary that maps integers to the characters
int2char = dict(enumerate(chars))

# Creating another dictionary that maps characters to integers
char2int = {char: ind for ind, char in int2char.items()}

print(char2int)


# + colab={"base_uri": "https://localhost:8080/"} id="6gyiJo1ISKik" outputId="9c93f0c3-b212-4826-9220-6d0878db0bbf"
def create_seq(text):
    length = 30
    sequences = list()
    for i in range(length, len(text)):
        if text[i-length-1] == ' ':
            # select sequence of tokens
            seq = text[i-length:i+1]
            # store
            sequences.append(seq)
    print('Total Sequences: %d' % len(sequences))
    return sequences

# create sequences   
sequences = create_seq(train_st)
sequences[:10]


# + colab={"base_uri": "https://localhost:8080/"} id="5VRa7gpwTDQM" outputId="aa0cb653-2737-4d68-9594-35ff1e9d6c11"
# create a character mapping index
# chars = sorted(list(set(data_new)))
# mapping = dict((c, i) for i, c in enumerate(chars))

def encode_seq(seq):
    sequences = list()
    for line in seq:
        # integer encode line
        encoded_seq = [char2int[char] for char in line]
        # store
        sequences.append(encoded_seq)
    return sequences

# encode the sequences
sequences = encode_seq(sequences)
sequences[0:1]

# + colab={"base_uri": "https://localhost:8080/"} id="yUxz4IhBTIVj" outputId="12976f64-d30b-43c1-893a-02b2aba91bec"
from sklearn.model_selection import train_test_split

# vocabulary size
vocab = len(char2int)
sequences = np.array(sequences)
# create X and y
X, y = sequences[:,:-1], sequences[:,-1]
# one hot encode y
y = to_categorical(y, num_classes=vocab)
# create train and validation sets
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

print('Train shape:', X_tr.shape, 'Val shape:', X_val.shape)

# + [markdown] id="WV0ayVzUTj1i"
# ## Building and Training the Model

# + colab={"background_save": true, "base_uri": "https://localhost:8080/"} id="_9L-ao5lTRHE" outputId="f8cacc49-5d0e-4d20-ec3e-6c1c82b044f3"
# define model
model = Sequential()
model.add(Embedding(vocab, 50, input_length=30, trainable=True))
model.add(GRU(150, recurrent_dropout=0.1, dropout=0.1))
model.add(Dense(vocab, activation='softmax'))
print(model.summary())

# compile the model
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
# fit the model
model.fit(X_tr, y_tr, epochs=30, verbose=2, validation_data=(X_val, y_val))


# + [markdown] id="rEw8uZxQTpfO"
# ## Predicting using the Model

# + colab={"background_save": true} id="jJYgc9JHTd1W"
# generate a sequence of characters with a language model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
    in_text = seed_text
    # generate a fixed number of characters
    for _ in range(n_chars):
        # encode the characters as integers
        encoded = [mapping[char] for char in in_text]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict character
        yhat = model.predict_classes(encoded, verbose=0)
        # reverse map integer to character
        out_char = ''
        for char, index in mapping.items():
            if index == yhat:
                out_char = char
                break
        # append to input
        in_text += char
    return in_text


# + colab={"background_save": true} id="JBfCzvVITgRI" outputId="b800a5de-4a18-48b2-f527-3919c8e33d6f"
for item in test_ls1:
    print([item, generate_seq(model, char2int, 30, item, 50)])

# + id="P7j0tcdcT5Lo"
# if you need to save the model to load somewhere else:
## model.save('Insert path here')
# if you need to load the model
## model = keras.models.load_model('Insert path here')
