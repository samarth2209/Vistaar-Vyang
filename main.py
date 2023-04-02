#Initiating the Libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

#Reading the JSON file
data = pd.read_json("Sarcasm_Headlines_Dataset.json", lines=True)
#print(data.head())

#Differentiating the Vistaar/Vyang
data["is_sarcastic"] = data["is_sarcastic"].map({0: "Vistaar", 1: "Vyang"})
#print(data.head())

#Setting 80/20 partion for training/testing
data = data[["headline", "is_sarcastic"]]
x = np.array(data["headline"])
y = np.array(data["is_sarcastic"])

cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#applying Naive-Bayes Model
model = BernoulliNB()
model.fit(X_train, y_train)
#print(model.score(X_test, y_test))

#Final Checking via user input
user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)


