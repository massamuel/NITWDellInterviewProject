from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import scipy
from sklearn.svm import SVC
from datetime import date, datetime
from sklearn.model_selection import GridSearchCV
import numpy as np
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

df = pd.read_csv(str(Path(__file__).resolve().parents[3] / "data/raw/mbti_1.csv"))

print("preprocess text")
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

print("training")
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
print("SGD score: {}%".format(int(sgd_score*100)))


param_grid = {'average': [True, False],
              'penalty': ['l1','l2']}

grid_search = GridSearchCV(clf,param_grid=param_grid)
grid_search.fit(X_train,y_train)

print(grid_search.best_params_)
best_hyper_params_sgd = list(grid_search.best_params_.items())

print(str(datetime.now()) + "Best Hyper Parameters to use for SGD Model : " + str(best_hyper_params_sgd))


log_message = str(datetime.now()) + " : Model SGD Classifier :  " + "SGD score: {}%".format(int(sgd_score*100))
print(log_message)
svm = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
svm_score = svm.score(X_test,y_test)

param_grid_svm = {'C': [1,0.5,0.2,0.4,2,1.5 ],
              'kernel': ['linear']}

grid_search_svm = GridSearchCV(clf,param_grid=param_grid)
grid_search_svm.fit(X_train,y_train)
print(grid_search_svm.best_params_)
# print("SVM Score: {}%".format(int(svm_score * 100)))
best_hyper_params_svm = list(grid_search_svm.best_params_.items())
print(str(datetime.now()) + "Best Hyper Parameters to use for SVC Model : " + str(best_hyper_params_svm))


log_message_svm = str(datetime.now()) + " : Model SVC Classifier :  " + "SVM Score: {}%".format(int(svm_score * 100))
print(log_message_svm)


print("Logistic Regression Training with cross fold validation")
targets = df.type.unique()
count = 0
while count < 16:  
    target_split = targets[count:count+2]
    temp_df = df[(df['type'] == target_split[0]) | (df['type'] == target_split[1])]
    vectorizer = TfidfVectorizer()
    encoder = LabelEncoder()
    new_X = vectorizer.fit_transform(temp_df['proccessed_posts'])
    new_y = encoder.fit_transform(temp_df['type'])
    log_reg = LogisticRegression()
    scores = cross_val_score(log_reg, new_X, new_y, cv=3)
    print("Cross Validation Score 5 folds For Labels {} and {}: {}".format(target_split[0],target_split[1], scores))
    print(str(datetime.now()) + "Cross Validation Score 5 folds For Labels {} and {}: {}".format(target_split[0],target_split[1], scores))
    
    count = count + 2

print("Done")
