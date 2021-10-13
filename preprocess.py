import pandas as pd 
import re
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark import SparkContext

# print(spark.version)

# df = spark.read.csv("mbti_1.csv")
# df.printSchema()
df = pd.read_csv("mbti_1.csv")

# sample = df.posts[0].split('|||')
# type = df.type[0]

documents_and_labels = []


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


docs = []
for i in range(len(df)):
    split_posts = df.posts[i].split('|||')
    for s in split_posts:
        sentence_split = [word for word in s if 'http' not in word]
        sentence_no_punct = [re.sub('[^a-zA-Z]',' ',text) for text in sentence_split]
        sentence_all_lower = [s.lower() for s in sentence_no_punct]


    # print(text_split)
    # if(len(sentence) > 0):
    #     sentence_no_blanks = [i for i in text_split if len(i) > 0]
    #     docs.append(sentence_no_blanks)
# avg_doc_length = sum([len(l) for l in docs]) / len(docs)

# docs_final = [' '.join(d).lower() for d in docs if len(d) > avg_doc_length]

# for doc in docs_final:
#     documents_and_labels.append({type : doc})



