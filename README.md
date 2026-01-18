# Customer Analytics: Regression & Clustering

This project demonstrates the application of regression and clustering techniques to a customer churn dataset using Python. +++ python .\regression_analysis.py +++
python .\data_clustering.py +++

---

## Task 1: Regression Analysis

**Objective:**  
To analyze the relationship between customer usage and billing using linear regression.

**Methodology:**
- Built a Linear Regression model
- Predicted **Total day charge** using **Total day minutes**
- Split data into training and testing sets (80/20)
- Evaluated the model using:
  - Mean Squared Error (MSE)
  - R-squared score (R²)
- Visualized results with a regression plot

**File:**  
`regression_analysis.py`

**Output:**  
- Regression performance metrics  
- Plot saved as `regression_result.png.`

---

## Task 3: Clustering Analysis

**Objective:**  
To segment customers based on usage and service behavior.

**Methodology:**
- Applied K-Means clustering
- Features used:
  - Total day minutes
  - Customer service calls
- Standardized features before clustering
- Identified 3 distinct customer groups
- Visualized clusters using a scatter plot

**File:**  
`data_clustering.py`

**Output:**  
- Cluster distribution summary  
- Visualization saved as `kmeans_clusters.png.`

---

## Tools & Libraries
- Python
- Pandas
- Matplotlib
- Scikit-learn

---

## Dataset
- Customer churn dataset (`churn-bigml-20.csv`)
