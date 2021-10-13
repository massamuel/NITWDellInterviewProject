import torch
import pandas as pd 
import scipy
import re
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

df = pd.read_csv("mbti_1.csv")

print("Preprocessing Text ")
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
        ##Extract sentences or questions with a length longer than average length
    corpus_no_links = ' '.join([' '.join(c) for c in corpus])
    text = re.sub('[^a-zA-Z]',' ',corpus_no_links)
    text = text.lower()
    documents.append(corpus_no_links)

df['proccessed_posts'] = documents


vectorizer = TfidfVectorizer()
encoder = LabelEncoder()
X = vectorizer.fit_transform(df['proccessed_posts'])
y = encoder.fit_transform(df['type'])
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.20, stratify=df['type'], random_state=42)

x_train = torch.tensor(scipy.sparse.csr_matrix.todense(X_train)).float()
x_test = torch.tensor(scipy.sparse.csr_matrix.todense(X_test)).float()
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

print("Model Training")
from torch import nn
model = nn.Sequential(
             nn.Linear(x_train.shape[1], 64),
             nn.ReLU(),
             nn.Linear(64, df['type'].nunique()),
             nn.LogSoftmax(dim=1))
# Define the loss
criterion = nn.NLLLoss()
# Forward pass, log  
logps = model(x_train)
# Calculate the loss with the logits and the labels
loss = criterion(logps, y_train)
loss.backward()
# Optimizers need parameters to optimize and a learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

print("Training phase")
epochs = 50
for e in range(epochs):
    optimizer.zero_grad()
    output = model.forward(x_train)
    loss = criterion(output, y_train)
    print(loss)
    loss.backward()
    optimizer.step()

print("Evaluation")
with torch.no_grad():
    model.eval()
    log_ps = model(x_test)
    test_loss = criterion(log_ps, y_test)
    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == y_test.view(*top_class.shape)
    test_accuracy = torch.mean(equals.float())
    print("Pytorch Accuracy: {}%".format(test_accuracy.item()))