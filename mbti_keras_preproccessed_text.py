import pandas as pd 
import keras 
import re
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer,hashing_trick
from keras.layers import Embedding, LSTM, Dropout, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
from log_metrics import log_accuracies


df = pd.read_csv('proccessed_text.csv')

targets = df.target.unique()
count = 0
while count < 16:  
    target_split = targets[count:count+2]
    print(target_split)
    temp_df = df[(df['target'] == target_split[0]) | (df['target'] == target_split[1])]
    tokenizer = Tokenizer()
    encoder = LabelEncoder()

    
    X = temp_df['text']
    tokenizer.fit_on_texts(X)
    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=1000)
    y = encoder.fit_transform(temp_df['target'])
    n_targets = len(np.unique(y))

    print(n_targets)
    maxlen = 1000
    vocab_size = len(tokenizer.word_index) + 1
    
    callback = EarlyStopping(monitor='loss', patience=3)
    model = keras.models.Sequential([
        keras.layers.Embedding(vocab_size,100,input_length=X.shape[1]),
        keras.layers.LSTM(128,dropout = 0.5),
        keras.layers.Dense(1, activation="sigmoid")

    ])


    n_targets = len(np.unique(y))
    print(n_targets)
    maxlen = 1000
    vocab_size = len(tokenizer.word_index) + 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    history = model.fit(X_train, y_train,batch_size=64, epochs=10, verbose=1, validation_split=0.2,callbacks=[callback])
    score = model.evaluate(X_test, y_test, verbose=1)
    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])

    print("TEST ACCURACY FOR CLASSES {}, {} : {}".format(target_split[0],target_split[1], score[1]))

    count = count + 2