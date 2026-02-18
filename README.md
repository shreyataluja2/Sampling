# Credit Card Fraud Detection: Sampling Technique Analysis

## Project Overview

This project presents a comprehensive experimental study on **credit card fraud detection** with a specific focus on the impact of **data sampling techniques** when dealing with **highly imbalanced datasets**. The primary objective is to analyze how different sampling strategies influence the performance of multiple machine learning models in identifying fraudulent transactions.

Fraud detection datasets typically suffer from severe class imbalance, where fraudulent transactions represent only a very small fraction of the total data. This project systematically evaluates whether sampling methods can mitigate this issue and improve classification performance.



## 1. Methodology

The project follows a structured and reproducible data science pipeline consisting of data preprocessing, sampling, model training, and performance evaluation.



### Data Preprocessing

* **Dataset Used:** `Creditcard_data (sampling).csv`
* **Initial Challenge:** The dataset was highly imbalanced, with non-fraudulent transactions (Class 0) overwhelmingly outnumbering fraudulent ones (Class 1).

#### Balancing Strategy

To address the imbalance issue, **Manual Random Over-Sampling** was applied to the minority (fraud) class until it matched the size of the majority (non-fraud) class. This resulted in a **balanced dataset containing 1,526 records**, enabling fair model evaluation.



### Sampling Techniques

After balancing, five distinct sampling techniques were applied to generate multiple training subsets:

1. **Simple Random Sampling**
   A random 20% subset selected from the balanced dataset.

2. **Systematic Sampling**
   Records selected at fixed intervals (every 2nd observation) to preserve ordering structure.

3. **Stratified Sampling**
   A 60% sample that maintains the original class distribution (Fraud vs Non-Fraud).

4. **Cluster Sampling**
   The dataset divided into 5 clusters, from which selected clusters were used for training.

5. **Bootstrap Sampling**
   Random sampling with replacement, producing a dataset equal in size to the balanced dataset.



### Machine Learning Models

Each sampled dataset was used to train and evaluate the following five classifiers:

* **M1:** Logistic Regression
* **M2:** Decision Tree
* **M3:** Random Forest
* **M4:** K-Nearest Neighbors (KNN, k = 1)
* **M5:** Support Vector Machine (SVM)

This setup resulted in a total of **25 model–sampling combinations**.



## 2. Accuracy Matrix (Results)

The table below summarizes the classification accuracy (%) achieved by each model under different sampling strategies:

| Model              | Simple Random | Systematic | Stratified | Cluster | Bootstrap |
| ------------------ | ------------- | ---------- | ---------- | ------- | --------- |
| M1 (Logistic)      | 95.08         | 91.50      | 90.71      | 89.43   | 95.42     |
| M2 (Decision Tree) | 98.36         | 99.35      | 98.91      | 98.37   | 99.35     |
| M3 (Random Forest) | 100.0         | 100.0      | 100.0      | 100.0   | 100.0     |
| M4 (KNN)           | 96.72         | 97.39      | 98.36      | 98.37   | 99.02     |
| M5 (SVM)           | 81.97         | 64.05      | 63.93      | 66.67   | 73.20     |



## 3. Results Analysis & Visualizations

### Performance Comparison

To better understand the results, two visualizations were generated:

* **`performance_comparison.png`**
  Compares the accuracy of each machine learning model (M1–M5) across all sampling techniques.

* **`technique_effectiveness.png`**
  Displays the average accuracy achieved by each sampling technique across all models.



### Key Observations

* **Top Performing Model:**
  Random Forest (M3) consistently achieved **100% accuracy** across all sampling methods, demonstrating exceptional robustness and stability.

* **Most Effective Sampling Methods:**
  Bootstrap Sampling and Simple Random Sampling produced the most reliable and consistently high performance across both linear and non-linear models.

* **Model Sensitivity:**
  Support Vector Machine (M5) exhibited a significant drop in accuracy under systematic and stratified sampling, indicating higher sensitivity to data ordering and distribution.

* **Tree-Based Models:**
  Decision Tree and Random Forest models were less affected by sampling variations, highlighting their suitability for fraud detection tasks.



## 4. Key Learnings

* Sampling techniques play a critical role in handling imbalanced datasets.
* Ensemble models like Random Forest are inherently robust to sampling variability.
* Model performance should always be evaluated in conjunction with data preparation strategies.
* Accuracy alone may not fully reflect model effectiveness in real-world fraud detection scenarios.



## 5. Conclusion

This project demonstrates that combining appropriate **sampling strategies** with robust **machine learning models** significantly improves fraud detection performance. While Random Forest emerged as the most reliable classifier, Bootstrap and Simple Random Sampling proved to be the most effective sampling techniques overall. The study highlights the importance of thoughtful data preparation in solving real-world imbalanced classification problems.


