import pandas as pd
import torch 
from tensorflow import keras 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import scipy
from sklearn.svm import SVC
from datetime import datetime
from log_metrics import log_accuracies


df = pd.read_csv("mbti_1.csv")


## Preprocess Text 
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
    documents.append(corpus_no_links)

df['proccessed_posts'] = documents


## Training
vectorizer = TfidfVectorizer()
encoder = LabelEncoder()
X = vectorizer.fit_transform(df['proccessed_posts'])
y = encoder.fit_transform(df['type'])
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.20, stratify=df['type'], random_state=42)

clf = SGDClassifier()
clf.fit(X_train,y_train)
sgd_score = clf.score(X_test,y_test)
accuracy_score = "SGD score: {}%".format(int(sgd_score*100))
# print("SGD score: {}%".format(int(sgd_score*100)))

log_message = datetime.now() + " : Model SGD Classifier :  " + "SGD score: {}%".format(int(sgd_score*100))
log_accuracies(log_message)

svm = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
svm_score = svm.score(X_test,y_test)
print("SVM Score: {}%".format(int(svm_score * 100)))

log_message_svm = datetime.now() + " : Model SVC Classifier :  " + "SVM Score: {}%".format(int(svm_score * 100))
log_accuracies(log_message_svm)


#Nural Net
import scipy


print("Done")