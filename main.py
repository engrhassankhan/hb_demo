import slate
import numpy as np
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB


import pandas as pd

df = pd.read_csv('resume_dataset.csv', delimiter=',')
print(df.shape)

print(df.head(5))


# importing all the words from the resumes into a list
df1 = df['Resume']

resume_list = df1.values.tolist()
# for i in range(97):
#     filename = "/c" + str(i+1) + ".pdf"
#     f = open("CVs" + filename, "rb")
#     doc = slate.PDF(f)
#     each_resume = ""
#     for j in range(len(doc)):
#         each_resume += doc[j]
#     resume_list.append(each_resume)
#
print(len(resume_list))
#
# print(len(resume_list[0]))
# # removing punctuations and other unnecessary characters
for i in range(len(resume_list)):


    table = str.maketrans(dict.fromkeys(string.punctuation))

    resume_list[i] = resume_list[i].translate(table)
    # s = s.translate(None, "\n")
    # s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', s)

print(len(resume_list))
# labelling the existing resumes as being accepted(1) or being rejected(0)
# The first 36 resumes are labelled as accepted in this case and a label list is prepared
label = []

#100 percentage matching candiate
#qualification, experience and cerifications and skills matachs.
for i in range(200):
    label.append(100)

#80 percentage matching candiate
#qualification, experience and cerifications and skills matachs.
for i in range(200):
    label.append(80)

#60 percentage matching candiate
#qualification, experience and cerifications and skills matachs.
for i in range(200):
    label.append(60)

#50 percentage matching candiate
#qualification, experience and cerifications and skills matachs.
for i in range(200):
    label.append(50)

#below 40 percentage matching candiate
#qualification, experience and cerifications and skills matachs.
for i in range(419):
    label.append(40)


label = np.array(label)


# shuffling and splitting the data into a training set and a testing set
resumes_train, resumes_test, y_train, y_test = train_test_split(resume_list, label, test_size=0.33, random_state=42)
#
# extracting words as features from the training and testing sets and making corresponding feature matrices
vectorizer = TfidfVectorizer(analyzer="word", stop_words="english", max_features=250)
features_train = vectorizer.fit_transform(resumes_train)
X_train = features_train.toarray()
features_test = vectorizer.fit_transform(resumes_test)
X_test = features_test.toarray()
#
# Using Decision Tree Classifier on the data
dtclf = tree.DecisionTreeClassifier()
dtclf = dtclf.fit(X_train, y_train)
print("Decision Tree Classifier")
print(dtclf.score(X_train, y_train)*100)
print(dtclf.score(X_test, y_test)*100)
# print("Random Forest Classifier")
# # Using Random Forest Classifier on the data
# rfclf = RandomForestClassifier()
# rfclf = rfclf.fit(X_train, y_train)
# print(rfclf.score(X_train, y_train)*100)
# print(rfclf.score(X_test, y_test)*100)
# print("SVM Classifier")
# # Using SVM Classifier on the data
# model_svm = svm.SVC()
# model_svm = model_svm.fit(X_train, y_train)
# print(model_svm.score(X_train, y_train)*100)
# print(model_svm.score(X_test, y_test)*100)
# print("Bernoulli Naive Bayes Algorithm")
# # Using Bernoulli Naive Bayes Algorithm
# bnbclf = BernoulliNB()
# bnbclf = bnbclf.fit(X_train, y_train)
# print(bnbclf.score(X_train, y_train)*100)
# print(bnbclf.score(X_test, y_test)*100)
# print("Gaussian Naive Bayes Algorithm")
# # Using Gaussian Naive Bayes Algorithm
# gnbclf = GaussianNB()
# gnbclf = gnbclf.fit(X_train, y_train)
# print(gnbclf.score(X_train, y_train)*100)
# print(gnbclf.score(X_test, y_test)*100)
#
# Testing a sample resume of a new applicant
# Replace "your_file_name" with the name of your resume doc and comment out the following lines of code

# print(X_test[0])

f = open("c10.pdf", "rb")
sample_resume = slate.PDF(f)
sample_resume = sample_resume[0]
sample_resume = vectorizer.transform([sample_resume])

print("The output of Testing", dtclf.predict(sample_resume)[0])




