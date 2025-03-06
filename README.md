# neural-network-challenge-1
Module Challenge 18
🎓 Student Loan Credit Ranking Neural Network
🚀 Project Overview
This project uses deep learning to predict student loan credit rankings based on various student metrics. It implements a neural network model to classify students into credit ranking categories (0 or 1) using TensorFlow and Keras.
📝 Description
The model analyzes various student attributes such as payment history, location, GPA ranking, study major, and financial aid scores to predict their credit ranking. The prediction helps determine the likelihood of successful loan repayment.
📊 Dataset
The dataset contains multiple features related to student loans:

💰 payment_history
📍 location_parameter
🔬 stem_degree_score
📈 gpa_ranking
🎓 alumni_success
📚 study_major_code
⏱️ time_to_completion
💵 finance_workshop_score
👥 cohort_ranking
💸 total_loan_score
🏦 financial_aid_score
⭐ credit_ranking (target variable)

🛠️ Resources and Software Used
💻 Programming Language

Python 3

🧪 Development Environment

Google Colab
Jupyter Notebook

📚 Libraries and Frameworks

🧠 TensorFlow 2.x - Deep learning framework
🔮 Keras - Neural network API
🐼 pandas - Data manipulation and analysis
🔢 NumPy - Numerical computing
🤖 scikit-learn - Machine learning tools

train_test_split - Data splitting
StandardScaler - Feature scaling
classification_report - Model evaluation



📂 Version Control

Git
GitHub

📁 Project Structure

📓 student_loans_with_deep_learning.ipynb - Jupyter notebook containing all code and analysis
💾 student_loans.keras - Saved neural network model
📄 README.md - Project documentation

🔍 Implementation Details
🧹 Data Preparation

📥 Load data from CSV
✂️ Split features and target variable
🔄 Split data into training and testing sets
⚖️ Scale features using StandardScaler

🏗️ Model Architecture

🧠 Sequential neural network
2️⃣ Two hidden layers with ReLU activation

First hidden layer: 16 neurons
Second hidden layer: 8 neurons


🎯 Output layer with sigmoid activation
Binary classification (credit ranking 0 or 1)

🏋️‍♀️ Model Training

📉 Binary crossentropy loss function
⚙️ Adam optimizer
📏 Accuracy evaluation metric
🔄 50 epochs for training

📊 Model Evaluation

✅ Accuracy score
📈 Classification report (precision, recall, f1-score)

🏆 Results
The model achieves approximately 73-75% accuracy on the test data, demonstrating its ability to predict student credit rankings based on the given features. Not bad for our first deep learning model! 🎉
🚀 Installation and Usage

Clone the repository:

Copygit clone https://github.com/yourusername/neural-network-challenge-1.git

Navigate to the project directory:

Copycd neural-network-challenge-1

Open the Jupyter notebook in Google Colab or locally:

Copyjupyter notebook student_loans_with_deep_learning.ipynb

Run the notebook cells in sequence to:

📥 Load and prepare the data
🧠 Build and train the neural network
📊 Evaluate the model
🔮 Make predictions
💾 Save the model



📜 License
This project is part of an educational bootcamp assignment.
👏 Acknowledgments

Dataset provided by edX Boot Camps LLC
Neural network implementation based on TensorFlow and Keras documentation