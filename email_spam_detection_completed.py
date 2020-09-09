import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV

##Step1: Load Dataset
dataframe = pd.read_csv("spam.csv")
# print(dataframe.describe())

##Step2: Split in to Training and Test Data

etext = dataframe["EmailText"]
label = dataframe["Label"]

etext_train, label_train = etext[0:4457],label[0:4457]
etext_test, label_test = etext[4457:],label[4457:]

##Step3: Extract Features
cv = CountVectorizer()  
features = cv.fit_transform(etext_train)

##Step4: Build a model
tuned_parameters = {'kernel': ['rbf','linear'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}

model = GridSearchCV(svm.SVC(), tuned_parameters)

model.fit(features,label_train)

print(model.best_params_)
#Step5: Test Accuracy
print("Accuracy is : ",model.score(cv.transform(etext_test), label_test))



