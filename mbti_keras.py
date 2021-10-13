from keras.backend import dropout
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
from datetime import datetime
from log_metrics import log_accuracies

df = pd.read_csv("mbti_1.csv")


## Preprocess Text 

# def preprocess_text(df,column,num_rows):
#     data_corpus = []
#     for i in range(num_rows):
#         text = re.sub('[^a-zA-Z]',' ',df[column][i])
#         text = text.lower()
#         text = text.split()
#         text = [word for word in text if not word in set(stopwords.words('english'))]
#         text = [word for word in text if not word not in set(nltk.corpus.words.words())]
#         text = [let for let in text if len(let) > 1]   
#         data_corpus.append(" ".join(text))
#     return data_corpus

print("preprocessing text")
documents = []
# exclude = set(string.punctuation)
for i in range(len(df)):
    first_instance = df.posts[i].split('|||')
    corpus = []
    for i in first_instance:
        sentence = [word for word in i.split(' ') if 'http' not in word]
        if(len(sentence) > 0):
            sentence_no_blanks = [i for i in sentence if len(i) > 0]
            corpus.append(sentence_no_blanks)
    
    corpus_no_links = ' '.join([' '.join(c) for c in corpus])
    text = re.sub('[^a-zA-Z]',' ',corpus_no_links)
    text = text.lower()
    documents.append(text)

df['proccessed_posts'] = documents


X = df['proccessed_posts'] 

tokenizer = Tokenizer()
encoder = LabelEncoder()
tokenizer.fit_on_texts(X)
y = encoder.fit_transform(df['type'])
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X, maxlen=100)

maxlen = 1000
vocab_size = len(tokenizer.word_index) + 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


print("model trainig")
callback = EarlyStopping(monitor='loss', patience=3)
model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size,100,
                           
                           input_length=100,
                           trainable = False),
    keras.layers.LSTM(128,return_sequences=True),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1, validation_split=0.2,callbacks=[callback])

#Storing model score from testing data 
score = model.evaluate(X_test, y_test, verbose=1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])


log_message_keras = str(datetime.now()) + " : Model Keras Classifier :  " + " Test Score: {}%, Test Accuracy: {}% ".format(int(score[0] * 100),int(score[1] * 100) )
log_accuracies(log_message_keras)

model2 = keras.models.Sequential([
    keras.layers.Embedding(vocab_size,100,
                           input_length=100,
                           trainable = False),
    keras.layers.LSTM(128,return_sequences=True),
    keras.layers.Dropout(0.5),
    keras.layers.LSTM(64,return_sequences=True),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(1, activation="sigmoid")
])

model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history2 = model2.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1, validation_split=0.2,callbacks=[callback])
score2 = model.evaluate(X_test, y_test, verbose=1)
print("Test Score:", score2[0])
print("Test Accuracy:", score2[1])


log_message_keras_model_2 = str(datetime.now()) + " : Model Keras Classifier 2 :  " + " Test Score: {}%, Test Accuracy: {}% ".format(int(score2[0] * 100),int(score2[1] * 100) )
log_accuracies(log_message_keras)