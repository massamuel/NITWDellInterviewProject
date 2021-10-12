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
score = clf.score(X_test,y_test)
print(score)

#Nural Net
import scipy

# x_train = torch.tensor(scipy.sparse.csr_matrix.todense(X_train)).float()
# x_test = torch.tensor(scipy.sparse.csr_matrix.todense(X_test)).float()
# y_train = torch.tensor(y_train)
# y_test = torch.tensor(y_test)


# from torch import nn
# model = nn.Sequential(
#              nn.Linear(x_train.shape[1], 64),
#              nn.ReLU(),
#              nn.Linear(64, df['type'].nunique()),
#              nn.LogSoftmax(dim=1))
# # Define the loss
# criterion = nn.NLLLoss()
# # Forward pass, log  
# logps = model(x_train)
# # Calculate the loss with the logits and the labels
# loss = criterion(logps, y_train)
# loss.backward()
# # Optimizers need parameters to optimize and a learning rate
# optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

print("Done")