# 📌 Spam Email Classifier using Logistic Regression

A machine learning project that classifies SMS messages as  **Spam**  or  **Ham**  using  **TF-IDF vectorization**  and  **Logistic Regression**.

----------

## 📂 Project Overview

This project demonstrates a complete workflow for building a text classification model:

-   Data loading & cleaning
    
-   Exploratory data analysis (EDA)
    
-   Text vectorization using  **TF-IDF**
    
-   Training a  **Logistic Regression**  model
    
-   Evaluating accuracy, precision, recall, and F1 score
    
-   Visualizing confusion matrix
    
-   Saving the trained model with  **joblib**
    

----------

## 📁 Dataset

The dataset used is the  **SMS Spam Collection Dataset**, containing two columns:

-   **v1**  →  `ham`  or  `spam`
    
-   **v2**  → SMS message text
    

The dataset is loaded using:

`data = pd.read_csv("/Users/adityasingh/codes/week 8 project/spam_email_classifier_updated/data/spam.csv", encoding='latin1')` 

----------

## 🔧 Technologies Used

-   **Python**
    
-   **NumPy**
    
-   **Pandas**
    
-   **Matplotlib / Seaborn**
    
-   **Scikit-learn**
    
-   **Joblib**
    

----------

## 📊 Exploratory Data Analysis (EDA)

Visualizing the distribution of spam vs ham messages:

`data['v1'].value_counts().plot(kind='bar', color=['blue', 'red'])
plt.title("Spam vs Ham Distribution")
plt.show()` 

----------

## 🧹 Preprocessing

### 1. Encode Labels

`from sklearn.preprocessing import LabelEncoder
data['label_num'] = LabelEncoder().fit_transform(data['v1'])` 

### 2. TF-IDF Feature Extraction

`tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(data['v2']).toarray()
y = data['label_num']` 

### 3. Train-Test Split

`X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 )` 

----------

## 🤖 Model Training

`model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)` 

----------

## 📈 Model Evaluation

`accuracy_score(y_test, y_pred)
classification_report(y_test, y_pred)
confusion_matrix(y_test, y_pred)` 

Confusion matrix heatmap:

`sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')` 

----------

## 💾 Saving the Model

`import joblib
joblib.dump(model, "spam_classifier_model.pkl")` 

This will allow reuse of the model without retraining.

----------

## 🚀 How to Run the Project

1.  Clone the repository
    
    `git clone <repo-url>` 
    
2.  Install dependencies
    
    `pip install -r requirements.txt` 
    
3.  Run the script
    
    `python spam_classifier.py` 
    
4.  Use the saved model for predictions
    
    `model = joblib.load("spam_classifier_model.pkl")` 
    

----------

## 📌 Future Improvements

-   Add Naive Bayes model for comparison
    
-   Deploy using Flask or FastAPI
    
-   Add real-time spam detection UI
    
-   Use word embeddings (Word2Vec, BERT)
    

----------

## 📝 Author

**Aditya Singh**  
Spam Email Classifier – Machine Learning Mini Project
