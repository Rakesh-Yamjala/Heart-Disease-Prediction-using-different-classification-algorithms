# **Heart Disease Prediction using Machine Learning**

## **Overview**
Heart disease remains one of the leading causes of mortality worldwide, and early detection is critical. This project implements and compares three popular machine learning classification algorithms:

- **K-Nearest Neighbors (KNN)**
- **Random Forest Classifier**
- **Decision Tree Classifier**

The goal is to identify the best-performing algorithm for heart disease prediction based on accuracy and other evaluation metrics.

---

## **Dataset**
The **Heart Disease UCI Dataset** is used for this project.

- **Dataset Source**: [Heart Disease UCI Dataset on Kaggle](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)
- **Number of Records**: 303
- **Number of Features**: 13 input features + 1 target feature (`target`)
- **Target Feature**:
  - `1`: Presence of heart disease
  - `0`: Absence of heart disease

### **Feature List**
- Age
- Sex
- Chest Pain Type (`cp`)
- Resting Blood Pressure (`trestbps`)
- Cholesterol
- Fasting Blood Sugar (`fbs`)
- Resting ECG (`restecg`)
- Max Heart Rate (`thalach`)
- Exercise-Induced Angina (`exang`)
- Oldpeak
- Slope
- Number of Major Vessels (`ca`)
- Thalassemia (`thal`)

---

## **Working Flow**

### **1. Data Preprocessing**
- Load the dataset.
- Handle missing values (if any) and standardize the data for KNN.
- Split the dataset into **Training (80%)** and **Testing (20%)** sets.

### **2. Model Implementation**
The following classification algorithms are implemented:

- **KNN**: Uses distance-based classification.
- **Decision Tree Classifier**: Builds a tree to model decision rules.
- **Random Forest Classifier**: Combines multiple decision trees to improve accuracy.

### **3. Hyperparameter Tuning**
- **GridSearchCV** is used for hyperparameter optimization:
  - Best `k` for **KNN**.
  - Best `max_depth` for **Decision Trees**.
  - Best `n_estimators` for **Random Forest**.

### **4. Model Evaluation**
The models are evaluated on the test set using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC**

### **5. Comparison**
Compare the results of all three models to identify the best-performing algorithm.

---

## **Results**
- **Random Forest Classifier** is the most reliable model for heart disease prediction due to its high accuracy and ability to handle complex data.
- **KNN** and **Decision Tree** can also be used for smaller datasets or when interpretability is required.

---

## **Technologies Used**
- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib / Seaborn

---

## **How to Run**
1. Clone the repository.
   ```bash
   git clone https://github.com/your-repo/heart-disease-prediction.git
   ```
2. Install required libraries.
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook or Python script.
   ```bash
   jupyter notebook heart_disease_prediction.ipynb
   ```

---

## **Conclusion**
This project demonstrates that the **Random Forest Classifier** achieves the best performance for heart disease prediction based on accuracy and other metrics. KNN and Decision Tree are also effective for smaller, less complex datasets.

---

## **Credits**
- **Dataset**: UCI Heart Disease Dataset (Kaggle)
- **Tools**: Python, Scikit-learn, Jupyter Notebook

