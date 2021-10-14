import pandas as pd 
import re
import numpy as np

df = pd.read_csv("mbti_1.csv")


import re
docs = []
for target in df.type.unique():
    df_class = df[df['type'] == target]
    df_class_changed_index = df_class.reset_index(drop=True)
    print(target)
    for i in range(len(df_class_changed_index)):
        split_posts = df_class_changed_index.posts[i].split('|||')
        corpus = []
        for s in split_posts:
            sentence_split = [word for word in s.split(' ') if 'http' not in word]
            sentence_no_punct = [re.sub('[^a-zA-Z]',' ',text) for text in sentence_split]
            sentence_all_lower = [s.lower() for s in sentence_no_punct]
            sentence_no_blanks = [s for s in sentence_all_lower if len(s) > 2]
            if(len(sentence_no_blanks) != 0):
                joined_sentence = ' '.join(sentence_no_blanks)
                corpus.append(joined_sentence)
        # print(corpus)
        avg_len = sum([len(c) for c in corpus]) / len(corpus)
        for c in corpus:
            if len(c) > avg_len:
                docs.append({"target": target, "text":c})

new_df = pd.DataFrame(docs)

print("NEW DATAFRAME CREATED WITH {} INSTANCES".format(new_df.shape[0]))

# new_df.to_csv('/Users/sampoplack/Desktop/NITWDell/project_repo/NITWDell/processed_text.csv')