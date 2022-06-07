import tensorflow as tf
import string
import requests

textdata= requests.get('https://www.gutenberg.org/cache/epub/67820/pg67820.txt')


textdata= textdata.text.replace('\r', '')
data= textdata.split('\n')

data= data[271:]

len(data)

data= " ".join(data)

def clean_text(file):
  token= file.split()
  table = str.maketrans('', '', string.punctuation)
  token = [w.translate(table) for w in token]
  token= [word for word in token if word.isalpha()]
  token = [word.lower() for word in token]
  return token

tokens= clean_text(data)
print(tokens[:50])

len(tokens)

len(set(tokens))

test_length = 50 + 1
lines=[]

for i in range(test_length, len(tokens)):
  seq= tokens[i- test_length:i]
  line = ' '.join(seq)
  lines.append(line)
print(len(lines))

seed_text= lines[12343]

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences= tokenizer.texts_to_sequences(lines)

sequences = np.array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]

vocab_len= len(tokenizer.word_index) + 1

y= to_categorical(y, num_classes= vocab_len)

seq_len= X.shape[1]

#LSTM MODEL
model= Sequential()
model.add(Embedding(vocab_len, 50, input_length=seq_len))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_len, activation='softmax'))
#summary of model
model.summary()

#compiling model
model.compile(loss= 'categorical_crossentropy', optimizer = 'adam', metrics= ('accuracy'))

model.fit(X, y, batch_size= 256, epochs = 100)

#generating text sequence
def text_generation(model, tokenizer, text_seq_len, seed_text, n_words):
  text=[]

  for _ in range(n_words):
    encoded= tokenizer.texts_to_sequences([seed_text])[0]
    encoded= pad_sequences([encoded], maxlen= text_seq_len, truncating='pre')

    y_predict= model.predict_classes(encoded)

    predicted_word= ''
    for word, i in tokenizer.word_index.items():
      if i== y_predict:
        predicted_word= word
        break
    seed_text= seed_text + ' '+ predicted_word
    text.append(predicted_word)
  return ' '.join(text)

text_generation(model, tokenizer, seq_len, seed_text, 100)