import pandas as pd 
import re
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark import SparkContext

# print(spark.version)

# df = spark.read.csv("mbti_1.csv")
# df.printSchema()
# df = pd.read_csv("mbti_1.csv")

# sample = df.posts[0].split('|||')
# type = df.type[0]

# documents_and_labels = []

# docs = []
# for i in sample:
#     sentence = [word for word in i.split(' ') if 'http' not in word]
#     sentence = ' '.join(sentence)
#     text = re.sub('[^a-zA-Z]',' ',sentence)
#     text = text.lower()
#     text_split = [t for t in text.split() if len(t) > 0]
#     # print(text_split)
#     if(len(sentence) > 0):
#         sentence_no_blanks = [i for i in text_split if len(i) > 0]
#         docs.append(sentence_no_blanks)
# avg_doc_length = sum([len(l) for l in docs]) / len(docs)

# docs_final = [' '.join(d).lower() for d in docs if len(d) > avg_doc_length]

# for doc in docs_final:
#     documents_and_labels.append({type : doc})



