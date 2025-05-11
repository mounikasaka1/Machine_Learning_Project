# Model Comparison Machine Learning Project 


This project applies and compares multiple machine learning classification models on different datasets and includes thorough exploratory data analysis (EDA), model tuning using GridSearchCV, and evaluation using standard classification metrics.


##  Project Structure

#add here 



## Dataset

#do other ones here 
The dataset (`FIC_Full.CSV`) contains anonymized medical records with features such as blood pressure, glucose level, BMI, etc., and a binary outcome indicating diabetes presence.

---

##  Exploratory Data Analysis (EDA)

EDA was conducted to better understand the dataset and includes:

* Data types and structure overview
* Summary statistics (mean, median, std, etc.)
* Distribution plots of key features
* Class balance analysis
* Correlation heatmap
* Outlier detection and duplicate checking

EDA visualizations are saved in the `eda_outputs/` directory.

---

##  Models Trained

Five classification algorithms were implemented and evaluated:

* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Classifier
* Support Vector Classifier (SVC)
* K-Nearest Neighbors (KNN)

Each model was trained using a pipeline with:

* Missing value imputation
* Feature scaling
* SMOTE for class balancing
* GridSearchCV for hyperparameter tuning

---

## Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC AUC Score

Results are compared and sorted by F1 Score.

---


##  How to Run

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook: `diabetes_classification_comparison.ipynb`

---

##  Dependencies

* `pandas`, `numpy`, `seaborn`, `matplotlib`
* `scikit-learn`
* `imblearn` (for SMOTE)

