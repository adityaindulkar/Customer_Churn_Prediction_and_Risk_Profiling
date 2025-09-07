# Customer Churn Prediction and Risk Profiling

This project developed a predictive churn model using machine learning (Random Forest, Gradient Boosting, and XGBoost) to identify high-risk customers. Feature engineering revealed key drivers including contract type, tenure, and service combinations, with month-to-month fiber optic customers showing 71.6% churn risk. The model achieved 81% accuracy with optimized thresholds to maximize recall for business implementation.

## Tech Stack: 
### Python (Numpy, Pandas, Scikit-Learn, Matplotlib)


## Features

- **Multiple ML Models**: Random Forest, Gradient Boosting, and XGBoost
- **Advanced Feature Engineering**: Binning and interaction features
- **Threshold Optimization**: Custom threshold tuning for optimal performance
- **Risk Profiling**: Identifies high-risk customer segments
- **Production Ready**: Pre-trained model loading for fast predictions.
- **Predictions**: Seperate predictions.py file to make predictions based on saved models and custom user inputs. 

## Model Performance Results

### Random Forest
- **Optimal Threshold**: 0.603
- **Accuracy**: 81%
- **Precision/Recall**: 0.62/0.71 (Churn class)
- **F1-Score**: 0.67 (Churn class)

### Gradient Boosting  
- **Optimal Threshold**: 0.377
- **Accuracy**: 81%
- **Precision/Recall**: 0.62/0.72 (Churn class)
- **F1-Score**: 0.66 (Churn class)

### XGBoost
- **Optimal Threshold**: 0.614
- **Accuracy**: 80%
- **Precision/Recall**: 0.60/0.71 (Churn class)
- **F1-Score**: 0.65 (Churn class)

## Key Insights

### Top Risk Factors:
1. **Contract Type**: Month-to-month customers are **15.1x** more likely to churn than 2-year contracts
2. **Tenure**: **53.5%** of all churn occurs in the first 12 months
3. **Service Combination**: Fiber optic + No tech support has **49.4%** churn rate
4. **High-Risk Profile**: Month-to-month + <12 months tenure + Fiber optic without tech support = **71.6%** churn rate

### Feature Importance:
**Most Important Features Across Models:**
- Contract type
- Tenure
- Fiber optic + No tech support combination
- Monthly/Total charges

### The system provides:
- **Probability Score**: 0-1 churn likelihood
- **Binary Prediction**: Churn/No Churn classification  
- **Risk Level**: High/Medium/Low categorization
- **Actionable Insight**: Immediate attention recommendations
  
### Example Output
Churn Probability: 0.788         
Prediction: Churn          
Risk Level: High
