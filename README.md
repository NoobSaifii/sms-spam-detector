# SMS Spam Detector 📱

A simple machine learning project to classify SMS messages as **spam** or **ham (not spam)** using **Naive Bayes** and the **Bag-of-Words** model.

---

## 📌 Features

* Loads and preprocesses the classic SMS Spam Collection dataset.
* Converts text messages into numerical features using **CountVectorizer**.
* Trains a **Naive Bayes (MultinomialNB)** classifier.
* Evaluates the model with **accuracy score** and **confusion matrix**.
* Allows users to test custom messages interactively.

---

## 🛠 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

**requirements.txt**:

```
pandas
scikit-learn
```

---

## 🚀 Usage

1. Clone this repository:

```bash
git clone https://github.com/YOUR_USERNAME/sms-spam-detector.git
cd sms-spam-detector
```

2. Run the script:

```bash
python spam_detector.py
```

3. Type your own message to test:

```text
pls type msg (or quit): Hello, how are you?
>> not spam msg

pls type msg (or quit): Congratulations! You won a free prize.
>> it spam msg
```

---

## 📂 Project Structure

```
sms-spam-detector/
│-- spam_detector.py        # Main script
│-- SMSSpamCollection       # Dataset (tab-separated: label, text)
│-- requirements.txt        # Dependencies
│-- README.md               # Project documentation
```

---

## 📊 Example Output

```
acc is: 0.97
confusion matrix:
[[965   3]
 [ 17 130]]
```

---

## 📖 Dataset

This project uses the **[SMS Spam Collection dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)** from UCI Machine Learning Repository.

---

## 📜 License

This project is licensed under the **MIT License** – feel free to use, modify, and share.
