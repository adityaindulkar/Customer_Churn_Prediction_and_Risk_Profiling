import pandas as pd
import numpy as np
import joblib

print("Loading pre-trained models...")
try:
    rf = joblib.load('random_forest_model.pkl')
    gb = joblib.load('gradient_boosting_model.pkl')
    xgb_model = joblib.load('xgboost_model.pkl')
    monthly_binner = joblib.load('monthly_binner.pkl')
    total_binner = joblib.load('total_binner.pkl')
    tenure_binner = joblib.load('tenure_binner.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    xgb_threshold = joblib.load('xgb_threshold.pkl')
    print("Models loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: Model files not found. {e}")
    print("Please run the training script first to generate model files.")
    exit()


def map_categorical_values(user_data):
    """Map categorical values to numerical format used in training"""
    
    mapping = {
        'gender': {'Male': 1, 'Female': 0},
        'SeniorCitizen': {'0': 0, '1': 1, '0.0': 0, '1.0': 1},
        'Partner': {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0},
        'Dependents': {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0},
        'PhoneService': {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0},
        'MultipleLines': {'Yes': 1, 'No': 0, 'No phone service': 2, 'yes': 1, 'no': 0},
        'InternetService': {'DSL': 0, 'Fiber optic': 2, 'No': 1},
        'OnlineSecurity': {'Yes': 1, 'No': 0, 'No internet service': 2, 'yes': 1, 'no': 0},
        'OnlineBackup': {'Yes': 1, 'No': 0, 'No internet service': 2, 'yes': 1, 'no': 0},
        'DeviceProtection': {'Yes': 1, 'No': 0, 'No internet service': 2, 'yes': 1, 'no': 0},
        'TechSupport': {'Yes': 1, 'No': 0, 'No internet service': 2, 'yes': 1, 'no': 0},
        'StreamingTV': {'Yes': 1, 'No': 0, 'No internet service': 2, 'yes': 1, 'no': 0},
        'StreamingMovies': {'Yes': 1, 'No': 0, 'No internet service': 2, 'yes': 1, 'no': 0},
        'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2, 'Month-to-Month': 0},
        'PaperlessBilling': {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0},
        'PaymentMethod': {
            'Electronic check': 0,
            'Mailed check': 1,
            'Bank transfer (automatic)': 2,
            'Credit card (automatic)': 3,
            'credit card': 3,
            'Credit card': 3,
            'Bank transfer': 2,
            'Credit card': 3
        }
    }
    
    mapped_data = user_data.copy()
    for feature, value_map in mapping.items():
        if feature in mapped_data:
            original_value = str(mapped_data[feature])
            lower_value = original_value.lower()
            for key in value_map:
                if lower_value == key.lower():
                    mapped_data[feature] = value_map[key]
                    break
            else:
                mapped_data[feature] = 0
    
    return mapped_data

def predict_churn(user_input):
    """Predict churn probability for a single customer"""
    
    # Create a DataFrame from user input
    user_df = pd.DataFrame([user_input])
    
    # DEBUG: Print what columns we have initially
    print(f"Input columns: {list(user_df.columns)}")
    
    # Apply preprocessing using the separate binners
    monthly_binned = monthly_binner.transform(user_df[['MonthlyCharges']])
    total_binned = total_binner.transform(user_df[['TotalCharges']])
    tenure_binned = tenure_binner.transform(user_df[['tenure']])
    
    # Add the binned columns
    user_df['MonthlyCharges_binned'] = monthly_binned
    user_df['TotalCharges_binned'] = total_binned
    user_df['Tenure_binned'] = tenure_binned
    
    # Drop original continuous columns
    user_df = user_df.drop(columns=['MonthlyCharges', 'TotalCharges', 'tenure'])
    
    # Create interaction features
    user_df['internet_tech'] = user_df['InternetService'].astype(str) + "_" + user_df['TechSupport'].astype(str)
    user_df = pd.get_dummies(user_df, columns=['internet_tech'])
    
    user_df['tenure_contract'] = user_df['Tenure_binned'].astype(str) + "_" + user_df['Contract'].astype(str)
    user_df = pd.get_dummies(user_df, columns=['tenure_contract'])
    
    # Drop the same columns as in training
    columns_to_drop = [
        'InternetService', 'TechSupport', 'internet_tech_0_2', 'internet_tech_1_1', 
        'internet_tech_1_0', 'internet_tech_2_1', 'tenure_contract_0.0_1', 
        'tenure_contract_0.0_2', 'tenure_contract_1.0_0', 'tenure_contract_1.0_1', 
        'tenure_contract_1.0_2', 'tenure_contract_2.0_0', 'tenure_contract_2.0_1', 
        'tenure_contract_2.0_2', 'tenure_contract_3.0_0', 'tenure_contract_3.0_1', 
        'tenure_contract_3.0_2', 'tenure_contract_4.0_0', 'tenure_contract_4.0_1', 
        'tenure_contract_4.0_2', 'tenure_contract_5.0_0', 'tenure_contract_5.0_1', 
        'tenure_contract_5.0_2', 'tenure_contract_6.0_0', 'tenure_contract_6.0_1', 
        'tenure_contract_6.0_2', 'tenure_contract_7.0_0', 'tenure_contract_7.0_1', 
        'tenure_contract_7.0_2', 'tenure_contract_8.0_0', 'tenure_contract_8.0_1', 
        'tenure_contract_8.0_2', 'tenure_contract_9.0_0', 'tenure_contract_9.0_1'
    ]
    
    for col in columns_to_drop:
        if col in user_df.columns:
            user_df = user_df.drop(columns=[col])
    
    # DEBUG: Check current columns vs expected
    current_cols = set(user_df.columns)
    expected_cols = set(feature_columns)
    print(f"Current columns: {current_cols}")
    print(f"Expected columns: {expected_cols}")
    print(f"Missing: {expected_cols - current_cols}")
    print(f"Extra: {current_cols - expected_cols}")
    
    # Ensure all columns match training data
    for col in feature_columns:
        if col not in user_df.columns:
            user_df[col] = 0
    
    # Drop any extra columns
    for col in user_df.columns:
        if col not in feature_columns:
            user_df = user_df.drop(columns=[col])
    
    # Reorder columns to match training data
    user_df = user_df[feature_columns]
    
    # Make prediction using XGBoost
    probability = xgb_model.predict_proba(user_df)[0, 1]
    prediction = (probability >= xgb_threshold).astype(int)
    
    return {
        'probability': probability,
        'prediction': 'Churn' if prediction == 1 else 'No Churn',
        'risk_level': 'High' if probability >= 0.7 else 'Medium' if probability >= 0.4 else 'Low'
    }

def get_user_input():
    """Collect user input for prediction"""
    print("\n" + "="*50)
    print("CUSTOMER CHURN PREDICTION INPUT")
    print("="*50)
    
    user_data = {
        'gender': input("Gender (Male/Female): ").strip(),
        'SeniorCitizen': input("Senior Citizen (0/1): ").strip(),
        'Partner': input("Partner (Yes/No): ").strip(),
        'Dependents': input("Dependents (Yes/No): ").strip(),
        'PhoneService': input("Phone Service (Yes/No): ").strip(),
        'MultipleLines': input("Multiple Lines (Yes/No/No phone service): ").strip(),
        'InternetService': input("Internet Service (DSL/Fiber optic/No): ").strip(),
        'OnlineSecurity': input("Online Security (Yes/No/No internet service): ").strip(),
        'OnlineBackup': input("Online Backup (Yes/No/No internet service): ").strip(),
        'DeviceProtection': input("Device Protection (Yes/No/No internet service): ").strip(),
        'TechSupport': input("Tech Support (Yes/No/No internet service): ").strip(),
        'StreamingTV': input("Streaming TV (Yes/No/No internet service): ").strip(),
        'StreamingMovies': input("Streaming Movies (Yes/No/No internet service): ").strip(),
        'Contract': input("Contract (Month-to-month/One year/Two year): ").strip(),
        'PaperlessBilling': input("Paperless Billing (Yes/No): ").strip(),
        'PaymentMethod': input("Payment Method (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic)): ").strip(),
        'MonthlyCharges': float(input("Monthly Charges: ").strip()),
        'TotalCharges': float(input("Total Charges: ").strip()),
        'tenure': int(input("Tenure (months): ").strip())
    }
    
    return user_data

def main_prediction():
    """Main function to predict churn from user input"""
    try:
        # Get user input
        user_input = get_user_input()
        
        # Map categorical values to numerical format
        mapped_input = map_categorical_values(user_input)
        
        # Make prediction
        result = predict_churn(mapped_input)
        
        # Display results
        print(f"\n{'='*50}")
        print("CHURN PREDICTION RESULTS")
        print(f"{'='*50}")
        print(f"Churn Probability: {result['probability']:.3f}")
        print(f"Prediction: {result['prediction']}")
        print(f"Risk Level: {result['risk_level']}")
        
        # Provide insights
        if result['probability'] >= 0.7:
            print("\n🔴 HIGH RISK CUSTOMER - Immediate attention needed!")
        elif result['probability'] >= 0.4:
            print("\n🟡 MEDIUM RISK CUSTOMER - Monitor closely")
        else:
            print("\n🟢 LOW RISK CUSTOMER - Stable customer")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("Please check your input format and try again.")

if __name__ == "__main__":
    main_prediction()