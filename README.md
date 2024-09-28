
# Fake News Detection Project - ADS final project

### By Vered Klein, Dana Markiewitz, and Sigal Grabois

---

## 🗂️ Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Goals](#-goals)
- [Features and Engineering](#-features-and-engineering)
  - [Statement Length](#1-statement-length)
  - [Doc2Vec Embeddings](#2-doc2vec-embeddings)
  - [Credibility Score](#3-credibility-score)
  - [TF-IDF](#4-tf-idf)
  - [Topic Modeling (LDA)](#5-topic-modeling-lda)
- [Modeling Approach](#-modeling-approach)
  - [Random Forest Classifier](#1-random-forest-classifier)
  - [Performance and Error Analysis](#2-performance-and-error-analysis)
- [Usage](#-usage)
- [Results](#-results)
- [Conclusion](#-conclusion)

---

## 📰 Overview

In today’s digital world, **fake news** is a pressing issue that can have far-reaching consequences. This project aims to tackle the problem by building and improving a machine learning model that predicts the **truthfulness** of statements. Using the **[LIAR dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)**, we explore how features like speaker credibility, statement length, and **Doc2Vec embeddings** contribute to detecting misinformation.

---

## 📊 Dataset

The **LIAR dataset** contains labeled statements with six levels of truthfulness:
- **True**
- **Mostly true**
- **Half true**
- **Barely true**
- **False**
- **Pants on fire**

Each statement includes metadata such as:
- **Speaker**: Who made the statement.
- **Job Title**: The speaker’s position.
- **State**: The state from which the speaker comes.
- **Party**: The political affiliation of the speaker.
- **Historical Count Labels**: Counts of previous statements labeled as true, false, etc.

---

## 🎯 Goals

1. **Predict the truthfulness** of statements based on the features.
2. **Use embeddings** (e.g., **Doc2Vec**) to capture the semantic meaning of statements and improve classification accuracy.
3. **Measure speaker credibility** using a custom credibility score based on their historical statements.
4. **Analyze correlations** between statement length and truthfulness.
5. **Apply topic modeling** to uncover patterns between true and false statements.

---

## ⚙️ Features and Engineering

### 1. **Statement Length**
   - **Word Count**: The number of words in a statement.
   - **Character Count**: The number of characters (excluding spaces) in the statement.
   - **Average Word Length**: Calculated for each statement.
   

### 2. **Doc2Vec Embeddings**
   - We use **Doc2Vec embeddings** to convert textual features like `subject`, `job_title`, and `context` into numerical vectors. This allows the model to capture semantic relationships.
  

### 3. **Credibility Score**
   - A speaker's credibility score is calculated by applying weighted penalties for false statements and rewarding truthful statements. This score reflects the speaker's overall reliability across their statements.
   
   ```python
   # Example of credibility score calculation
   credibility_score = 100 - (total_penalty / total_statements)
   ```

### 4. **TF-IDF**
   - **TF-IDF** (Term Frequency-Inverse Document Frequency) is used to assess the importance of words within a statement, helping the model weigh frequent and rare words appropriately.
  

### 5. **Topic Modeling (LDA)**
   - We apply **Latent Dirichlet Allocation (LDA)** to group statements into topics and explore whether certain topics are associated more with truth or falsehood.
  

---

## 🛠️ Modeling Approach

### 1. **Random Forest Classifier**
   - Our main model is the **Random Forest Classifier**, chosen for its ability to handle both numerical and categorical data.
   - **Hyperparameter Tuning**:
     - `n_estimators`: 300
     - `max_depth`: 20
     - `min_samples_split`: 5
     - `min_samples_leaf`: 15
     
     ```python
     from sklearn.ensemble import RandomForestClassifier
     model = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=5, min_samples_leaf=15)
     model.fit(X_train, y_train)
     ```

### 2. **Performance and Error Analysis**
   - **Confusion Matrix**: We perform error analysis by visualizing the confusion matrix and tracking which labels are most commonly misclassified.
   

---

## 🖥️ Usage

### 1. **Dependencies**
Ensure you have the following libraries installed:
```bash
pip install pandas numpy matplotlib seaborn gensim scikit-learn nltk wordcloud pyLDAvis
```

### 2. **Data Loading and Preprocessing**

Load the dataset:
```python
import pandas as pd
header = ['statement_id', 'label', 'statement', 'subject', 'speaker', 'job_title', 'state', 'party', 'barely_true_c', 'false_c', 'half_true_c', 'mostly_true_c', 'pants_on_fire_c', 'context']
train_data = pd.read_csv('train.tsv', sep='	', names=header)
valid_data = pd.read_csv('valid.tsv', sep='	', names=header)
test_data = pd.read_csv('test.tsv', sep='	', names=header)
```

### 3. **Run the Model**

```python
from sklearn.ensemble import RandomForestClassifier
# Train the model
model = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=5, min_samples_leaf=15)
model.fit(X_train_combined, y_train)

# Evaluate on test data
all_predictions = model.predict(X_test_combined)
```

### 4. **Visualizations**

To visualize the **topic modeling**:
```python
import pyLDAvis.gensim_models as gensimvis
lda_display = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(lda_display)
```

---

## 🏆 Results

- **Accuracy**: Our best accuracy on the test set was approximately **85%**.
- **Important Features**:
  - **TF-IDF**
  -  **Credibility Score**
  - **Statement Length**

---

## 🔍 Conclusion

This project successfully demonstrates how machine learning can be applied to detect fake news. By engineering features like credibility scores, using text embeddings, and employing **Random Forest Classifiers**, we achieved promising results in classifying statements based on their truthfulness. **Topic modeling** provided additional insights into patterns between truthful and false statements.

---
