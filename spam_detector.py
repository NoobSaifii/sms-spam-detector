import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# load dataset file
data = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["label","text"])

# make label number coz model not understand word
data["label_num"] = data.label.map({"ham":0, "spam":1})

# split train test
x_train, x_test, y_train, y_test = train_test_split(
    data["text"], data["label_num"], test_size=0.2, random_state=1
)

# change text to number using bag of word
cv = CountVectorizer()
x_train_cv = cv.fit_transform(x_train)
x_test_cv = cv.transform(x_test)

# make model naive bayes (easy, simple)
model = MultinomialNB()
model.fit(x_train_cv, y_train)

# test how good
pred = model.predict(x_test_cv)
print("acc is:", accuracy_score(y_test, pred))
print("confusion matrix:")
print(confusion_matrix(y_test, pred))

# now test  msg
while True:
    txt = input("pls type msg (or quit): ")
    if txt.lower() == "quit":
        break
    txt_cv = cv.transform([txt])
    out = model.predict(txt_cv)[0]
    if out == 1:
        print(">> it spam msg")
    else:
        print(">> not spam msg")
