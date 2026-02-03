

# 🛒 Sentiment Analysis of Flipkart Product Reviews

This project focuses on analyzing **real-time Flipkart product reviews** and classifying them as **Positive** or **Negative** using Natural Language Processing (NLP) and Machine Learning techniques.

The application allows users to enter a product review and instantly receive sentiment feedback through a simple web interface.

---

## 🎯 Project Objective

The main objectives of this project are:

- To classify customer reviews into **positive** and **negative** sentiments  
- To understand common pain points expressed in negative reviews  
- To identify factors contributing to customer satisfaction  
- To deploy a trained sentiment analysis model as a real-time web application  

---

## 📦 Dataset Description

The dataset consists of **8,518 customer reviews** collected from Flipkart for the product:

**YONEX MAVIS 350 Nylon Shuttle**

Each review contains the following attributes:

- Reviewer Name  
- Rating  
- Review Title  
- Review Text  
- Place of Review  
- Date of Review  
- Up Votes  
- Down Votes  

The dataset was pre-scraped and provided, and no additional scraping was performed.

---

## 🧹 Data Preprocessing

The following preprocessing steps were applied to the review text:

- Conversion to lowercase  
- Removal of special characters and punctuation  
- Removal of stopwords  
- Lemmatization to reduce words to their base form  

This helped in improving model performance and reducing noise in text data.

---

## 🔢 Feature Engineering

Text data was converted into numerical format using:

- **TF-IDF (Term Frequency – Inverse Document Frequency)**  

TF-IDF was chosen because it effectively captures the importance of words while reducing the impact of commonly occurring terms.

---

## 🤖 Model Training & Selection

Multiple machine learning algorithms were experimented with, including:

- Naive Bayes  
- Support Vector Machine (SVM)  
- Logistic Regression  

After evaluation, **Logistic Regression** was selected as the final model due to its superior performance.

### 📊 Evaluation Metric
- **F1-Score** was used as the primary evaluation metric to balance precision and recall.

---

## 🏆 Final Model

- **Model Used:** Logistic Regression  
- **Vectorizer:** TF-IDF  
- **Model Persistence:** Pickle (`.pkl`)  

The trained model and vectorizer were saved together for deployment.

---

## 🌐 Model Deployment

The trained sentiment classification model has been deployed using:

- **Streamlit** for web application development  
- **Hugging Face Spaces** for cloud hosting  

The deployed application provides a public interface where users can enter product reviews and receive real-time sentiment predictions.

---

## 🚀 Live Application

🔗 **Deployed App Link:**  
👉 https://huggingface.co/spaces/SriRam8765/SentimentAnalysis
---

## 🧪 How to Use the Application

1. Open the deployed link  
2. Enter a product review in the text box  
3. Click on **Predict Sentiment**  
4. The application will display whether the review is **Positive** or **Negative**

---

## 🛠 Technologies Used

- Python  
- Pandas & NumPy  
- NLTK  
- Scikit-Learn  
- Streamlit  
- Hugging Face Spaces  

---

## ✅ Project Outcome

- Successfully classified customer reviews into positive and negative sentiments  
- Built an end-to-end sentiment analysis pipeline  
- Deployed a real-time sentiment analysis web application  
- Enabled easy access to sentiment insights for users  

---

## 📌 Conclusion

This project demonstrates a complete machine learning workflow starting from data preprocessing and model training to real-time deployment. The deployed application can help businesses understand customer feedback and improve product quality based on sentiment insights.

---

## 👤 Author

**Sri Ram**

