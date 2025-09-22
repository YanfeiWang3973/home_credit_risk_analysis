# Home Credit Default Risk – ML Case Study


This project is an end-to-end machine learning pipeline built to analyze and predict credit default risk using the Home Credit Default Risk dataset
.

Because of Google Colab’s limited memory, the dataset was downsized to 20,000 rows, and all merging/feature engineering steps are based on this subset.

📌 Project Steps

Data Preparation

Sampled 20,000 rows from the full dataset for memory efficiency

Merged multiple related tables (application train + others)

Removed duplicates and cleaned missing values

Cleaned special characters in column names

Feature Engineering & Encoding

Categorical encoding (One-Hot and Label Encoding where needed)

Handled imbalanced target classes using upsampling and class_weight

Model Building

Random Forest Classifier

Parameters tuned (n_estimators, max_depth, class_weight)

LightGBM Classifier

Histogram-based boosting for speed and efficiency

Tuned with learning_rate, n_estimators, and class_weight

Model Evaluation

Predictions (.predict) and probabilities (.predict_proba)

Compared against true labels (y_test)

Evaluated with ROC and precision-recall curves

🛠️ Tech Stack

Python (Google Colab)

Pandas / NumPy (data wrangling)

Scikit-learn (Random Forest, metrics)

LightGBM (gradient boosting model)

📊 Results

Successfully built and compared Random Forest vs. LightGBM models

Demonstrated handling of imbalanced data and importance of probability-based evaluation

Showed practical tradeoffs between model interpretability and efficiency
