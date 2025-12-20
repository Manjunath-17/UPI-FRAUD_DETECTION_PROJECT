🔐 UPI Fraud Detection Using Machine Learning
📌 Project Overview

The Unified Payments Interface (UPI) has become one of the most widely used digital payment systems in India. However, the rapid growth of UPI transactions has also led to an increase in fraudulent activities such as phishing, unauthorized transactions, fake UPI requests, and social engineering attacks.
This project presents a machine learning–based fraud detection system that classifies UPI transactions as Fraudulent or Genuine by analysing transaction patterns and behavioural features.

The system uses multiple supervised machine learning algorithms and selects the best-performing model for real-time fraud prediction through a web interface.

🎯 Objectives

Detect fraudulent UPI transactions using machine learning

Analyse transaction behaviour and patterns

Compare multiple ML algorithms and select the best model

Provide real-time fraud prediction through a web interface

Enhance digital payment security and user trust

🧠 Machine Learning Models Used

Random Forest Classifier

Logistic Regression

Decision Tree Classifier

Support Vector Machine (SVM)

📌 Final Model Selected: Random Forest (based on performance comparison)

📊 Dataset Description

The project uses a synthetic UPI transaction dataset containing features such as:

Transaction amount

Transaction time (hour, day, month, year)

UPI number / identifier

Transaction-related attributes

Fraud label (0 = Genuine, 1 = Fraud)

⚙️ Methodology

Data collection and preprocessing

Feature selection and encoding

Train-test split (70% training, 30% testing)

Model training and evaluation

Performance comparison using metrics

Deployment using Flask web application

📈 Evaluation Metrics

The models are evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Random Forest achieved the best balance between precision and recall, making it suitable for deployment.

🖥️ System Architecture

Machine learning model trained using Python and Scikit-learn

Flask backend for model prediction

HTML & CSS frontend for user interaction

Real-time prediction of transaction status (Fraud / Genuine)

🌐 Web Interface Features

User-friendly transaction input form

Real-time fraud prediction

Clear visual indication of results:

⚠️ Fraud Detected

✅ Genuine Transaction

🛠️ Technologies Used

Programming Language: Python
Framework: Flask
ML Library: Scikit-learn
Frontend: HTML, CSS
Tools: VS Code, GitHub

🚀 How to Run the Project

Clone the repository

git clone https://github.com/your-username/upi-fraud-detection-ml.git


Install required libraries

pip install -r requirements.txt


Run the Flask application

python app.py


Open browser and go to:

http://127.0.0.1:5000/

🔮 Future Scope

Integration with real-time UPI payment APIs

Use of deep learning models (LSTM, CNN)

Risk score–based fraud prediction

Deployment on cloud platforms

Training on real-world large-scale datasets

📄 Project Status

✅ Completed
🎓 Academic Final Year Project

👨‍💻 Author

Manjunath Khot
Final Year Project – UPI Fraud Detection Using Machine Learning
