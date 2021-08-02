import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

##Step1: Load Dataset
dataframe = pd.read_csv("spam.csv")
print(dataframe.describe())

##Step2: Split in to Training and Test Data

x = dataframe["EmailText"]
y = dataframe["Label"]

x_train,y_train = x[0:4457],y[0:4457]
x_test,y_test = x[4457:],y[4457:]

##Step3: Build model
cv = CountVectorizer()
features = cv.fit_transform(x_train)

model = svm.SVC()
model.fit(features,y_train)

#Step4: Test Accuracy
features_test = cv.transform(x_test)
print(model.score(features_test, y_test))
