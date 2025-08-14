# Credit Card Fraud Detection

This project aims to detect fraudulent credit card transactions using machine learning.    
It compares the performance of **Logistic Regression** (a linear model) and **Random Forest** (an ensemble model) on a balanced dataset derived from the highly imbalanced credit card fraud dataset.

---

## 📂 Dataset

The dataset used in this project is from Kaggle:  
[Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

**Key details:**
- Contains transactions made by European cardholders in September 2013.
- Features are anonymized (V1, V2, …, V28) due to confidentiality.
- The target variable:
  - `0` → Non-fraud
  - `1` → Fraud
- Highly imbalanced: only ~0.17% of transactions are fraud.

---

## ⚙️ Project Workflow

### 1. Import Libraries
- **Pandas**, **NumPy**: Data handling
- **Scikit-learn**: Model training, scaling, and evaluation
- **Matplotlib**, **Seaborn**: Visualization

### 2. Load & Explore Data
- View first few rows
- Check data types and summary statistics
- Check for missing values

### 3. Handle Class Imbalance
- Downsample the majority class (`Class = 0`)
- Create a balanced dataset with equal fraud and non-fraud samples

### 4. Preprocess Data
- Split into features (X) and target (y)
- Train-test split (70%-30%)
- Standardize features using **StandardScaler**

### 5. Model Training
- **Logistic Regression**
- **Random Forest Classifier**

### 6. Model Evaluation
- Accuracy
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix Heatmap

---

## 📊 Results

### Logistic Regression
- Good baseline model
- Accuracy and classification metrics reported

### Random Forest
- Generally better performance than Logistic Regression
- Captures non-linear patterns in data

---

## 🖼️ Example Confusion Matrix
Logistic Regression Example:

|            | Predicted: Non-Fraud | Predicted: Fraud |
|------------|----------------------|------------------|
| **Actual: Non-Fraud** | True Negative         | False Positive  |
| **Actual: Fraud**     | False Negative        | True Positive   |

Heatmaps are plotted for both models.

---

## 🚀 How to Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Download Dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the project directory.

### 4️⃣ Run the Script
```bash
python fraud_detection.py
```

---

## 📦 Requirements

- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  

Install with:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 📌 Future Improvements
- Use SMOTE for upsampling instead of downsampling
- Try advanced algorithms like XGBoost or LightGBM
- Deploy as a web app with Flask or Streamlit

---

## 🙌 Acknowledgements
Dataset provided by [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud).
