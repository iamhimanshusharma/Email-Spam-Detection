import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV

##Step1: Load Dataset

df = pd.read_csv("spam.csv")
# print(df.head())

##Step2: Split in to Training and Test Data

etext = df["EmailText"]
label = df["Label"]

etext_train, label_train = etext[0:4457], label[0:4457]
etext_test , label_test = etext[4457:], label[4457:]

##Step3: Extract Features

cv  =  CountVectorizer()
features = cv.fit_transform(etext_train)

##Step4: Build a model

model = svm.SVC()
model.fit(features, label_train)


##Step5: Test Accuracy

features_test = cv.transform(etext_test)
print("Accuracy is : ",model.score(features_test, label_test))


