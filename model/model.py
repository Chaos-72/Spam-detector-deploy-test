import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import pickle


df = pd.read_csv("spam.csv", encoding='latin-1')
df.head()

# delete the un-neccessary columns
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# Convert spam -> 1 and ham -> 0
encoder = LabelEncoder()
df['spam'] = encoder.fit_transform(df['v1'])

# split the dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(df['v2'], df['spam'], test_size=0.3, random_state=42)

# Conver the text into vectors
cv = CountVectorizer()
x_train_cv = cv.fit_transform(x_train.values)

# Intialize model
model = MultinomialNB()

# train the model
model.fit(x_train_cv, y_train)

# Save model as 'nlp_model'
with open("nlp_model", "wb") as f:
    pickle.dump({
        "model": model,
        "vectorizer": cv,
        "target_names": ['ham', 'spam'], 
    }, f)
                 

print("Saved model along with 'vectorizer' and 'target_names' 'nlp_model'")
