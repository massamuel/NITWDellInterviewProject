import torch
import pandas as pd 
import scipy
import re

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
    text = re.sub('[^a-zA-Z]',' ',corpus_no_links)
    text = text.lower()
    documents.append(corpus_no_links)

df['proccessed_posts'] = documents

x_train = torch.tensor(scipy.sparse.csr_matrix.todense(X_train)).float()
x_test = torch.tensor(scipy.sparse.csr_matrix.todense(X_test)).float()
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)


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

epochs = 50
for e in range(epochs):
    optimizer.zero_grad()
    output = model.forward(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    model.eval()
    log_ps = model(x_test)
    test_loss = criterion(log_ps, y_test)
    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == y_test.view(*top_class.shape)
    test_accuracy = torch.mean(equals.float())
    print(test_accuracy)