# Customer Churn Analysis

## Objective
Analyze customer data to identify churn patterns and predict customer behavior.

## Tools Used
- Python (Pandas, Seaborn)
- Machine Learning (Logistic Regression)
- Power BI

## Features
- Data Cleaning
- Data Visualization
- Churn Prediction Model
- Interactive Dashboard

## Key Insights
- Month-to-month contracts have highest churn
- Higher charges lead to churn
- Long-term customers are more loyal

  ## Results
- Accuracy: 73%
- ROC-AUC: 0.83
- Recall (Churn class): 79% — catches 4 out of 5 customers who actually churn
- Precision (Churn class): 50%

## Why recall over accuracy
Missing an actual churner costs more than a false alarm, so the model was
tuned to prioritize recall for the churn class over raw accuracy.

## Conclusion
This project helps businesses reduce churn by identifying high-risk customers.
