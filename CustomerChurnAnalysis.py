import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, precision_recall_curve, confusion_matrix
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns
from matplotlib.patches import Circle
import joblib


df = pd.read_csv('Telco-Customer-Churn.csv')  
df = df.drop(columns=['customerID'])

n_bins = 10  
# Create separate binners for each feature
monthly_binner = KBinsDiscretizer(
    n_bins=n_bins,
    encode='ordinal',    
    strategy='quantile', 
    subsample=200000  
)
total_binner = KBinsDiscretizer(
    n_bins=n_bins,
    encode='ordinal',    
    strategy='quantile', 
    subsample=200000  
)
tenure_binner = KBinsDiscretizer(
    n_bins=n_bins,
    encode='ordinal',    
    strategy='quantile', 
    subsample=200000  
)

# Fit and transform each feature with its own binner
df['MonthlyCharges_binned'] = monthly_binner.fit_transform(df[['MonthlyCharges']])
df['TotalCharges_binned'] = total_binner.fit_transform(df[['TotalCharges']])
df['Tenure_binned'] = tenure_binner.fit_transform(df[['tenure']])
df = df.drop(columns=['MonthlyCharges', 'TotalCharges', 'tenure'])

df['internet_tech'] = df['InternetService'].astype(str) + "_" + df['TechSupport'].astype(str)
df = pd.get_dummies(df, columns=['internet_tech'])
df['tenure_contract'] = df['Tenure_binned'].astype(str) + "_" + df['Contract'].astype(str)
df = pd.get_dummies(df, columns=['tenure_contract'])

df = df.drop(columns=['InternetService', 'TechSupport', 'internet_tech_0_2', 'internet_tech_1_1', 'internet_tech_1_0', 'internet_tech_2_1', 'tenure_contract_0.0_1', 'tenure_contract_0.0_2', 'tenure_contract_1.0_0', 'tenure_contract_1.0_1', 'tenure_contract_1.0_2', 'tenure_contract_2.0_0', 'tenure_contract_2.0_1', 'tenure_contract_2.0_2', 'tenure_contract_3.0_0', 'tenure_contract_3.0_1', 'tenure_contract_3.0_2', 'tenure_contract_4.0_0', 'tenure_contract_4.0_1', 'tenure_contract_4.0_2', 'tenure_contract_5.0_0', 'tenure_contract_5.0_1', 'tenure_contract_5.0_2', 'tenure_contract_6.0_0', 'tenure_contract_6.0_1', 'tenure_contract_6.0_2', 'tenure_contract_7.0_0', 'tenure_contract_7.0_1', 'tenure_contract_7.0_2', 'tenure_contract_8.0_0', 'tenure_contract_8.0_1', 'tenure_contract_8.0_2', 'tenure_contract_9.0_0', 'tenure_contract_9.0_1']) 
# internet_tech_2_0 is among important features, so we keep it. so is tenure_contract_9.0_2 and tenure_contract_0.0_0
print(df.head())
X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. RF Predictions Model

rf = RandomForestClassifier(
    n_estimators=150,
    class_weight='balanced',
    max_depth=10,
    min_samples_leaf=5,
    random_state=42
)
rf.fit(X_train, y_train)

rf_probs = rf.predict_proba(X_test)[:, 1]

def optimize_threshold(y_true, probs, min_recall=0.7):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    viable_idx = np.where(recall >= min_recall)[0]
    optimal_idx = viable_idx[np.argmax(precision[viable_idx])] if len(viable_idx) > 0 else np.argmax(recall)
    return max(thresholds[optimal_idx], 0.2)

rf_threshold = optimize_threshold(y_test, rf_probs, min_recall=0.7)
rf_preds = (rf_probs >= rf_threshold).astype(int)

# 2. GRADIENT BOOSTING MODEL 

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_leaf=10,
    random_state=42
)
gb.fit(X_train, y_train)

gb_probs = gb.predict_proba(X_test)[:, 1]
gb_threshold = optimize_threshold(y_test, gb_probs, min_recall=0.7)
gb_preds = (gb_probs >= gb_threshold).astype(int)

# 3. XGBoost Model

scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,  
    eval_metric='aucpr',               
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)

xgb_probs = xgb_model.predict_proba(X_test)[:, 1]  
def xgb_optimize_threshold(y_true, probs, min_recall=0.7):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    viable_idx = np.where(recall >= min_recall)[0]
    if len(viable_idx) > 0:
        optimal_idx = viable_idx[np.argmax(precision[viable_idx])]
        return thresholds[optimal_idx]
    return 0.5  # Fallback

xgb_threshold = xgb_optimize_threshold(y_test, xgb_probs, min_recall=0.7)
xgb_preds = (xgb_probs >= xgb_threshold).astype(int)

# 4. MODEL COMPARISON

print("\n" + "="*40)
print("RANDOM FOREST PERFORMANCE")
print("="*40)
print(f"Optimal Threshold: {rf_threshold:.3f}")
print(classification_report(y_test, rf_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))

print("\n" + "="*40)
print("GRADIENT BOOSTING PERFORMANCE")
print("="*40)
print(f"Optimal Threshold: {gb_threshold:.3f}")
print(classification_report(y_test, gb_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, gb_preds))

print("\n" + "="*40)
print("XGBOOST PERFORMANCE")
print("="*40)
print(f"Optimal Threshold: {xgb_threshold:.3f}")
print(classification_report(y_test, xgb_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, xgb_preds))

# Feature Importance
xgb_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

rf_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)
gb_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': gb.feature_importances_
}).sort_values('Importance', ascending=False)
print("\n" + "="*40)
print("FEATURE IMPORTANCE")
print("="*40)
print("\nTop 10 Features from Random Forest:")
print(rf_importance.head(10))   
print("\nTop 10 Features from Gradient Boosting:")
print(gb_importance.head(10))
print("\nTop 10 Features from XGBoost:")
print(xgb_importance.head(10))

# =====================
# 4. STATISTICAL ANALYSIS
# =====================

print("\n" + "="*40)
print("CONTRACT TYPE CHURN RISK")
contract_churn_rate = df.groupby('Contract')['Churn'].mean()
print(contract_churn_rate.head())
# Month-to-month vs 1-year comparison and 2 Year comparison
risk_ratio1 = contract_churn_rate[0] / contract_churn_rate[1]
risk_ratio2 = contract_churn_rate[0] / contract_churn_rate[2]
print(f"Month-to-month customers are {risk_ratio1:.1f}x more likely to churn than 1-year customers and {risk_ratio2:.1f}x more likely than 2-year customers.")

print("\n" + "="*40)
print("TENURE BINS CHURN DISTRIBUTION")
tenure_churn0 = df.groupby('Tenure_binned')['Churn'].mean()
print(tenure_churn0.head())
# Churn Rate of Tenure 0 to 12 months
bin_0_to_2_churn_rate = df[df['Tenure_binned'].isin([0,1,2])]['Churn'].mean() * 100
print(f"Average churn rate in 0 to 12 months is: {bin_0_to_2_churn_rate:.1f}%")
# Percentage of all churners coming from customer tenure 0 to 12 months
total_churners = df[df['Churn'] == 1].shape[0]
churners_in_bins_0_to_2 = df[(df['Tenure_binned'].isin([0,1,2])) & (df['Churn'] == 1)].shape[0]
bin_0_to_2_churners_pct = (churners_in_bins_0_to_2 / total_churners) * 100
print(f"Percentage of all churners coming from customer tenure 0 to 12 months is: {bin_0_to_2_churners_pct:.1f}%")
# Churn distribution by tenure bins
tenure_churn = df[df['Churn']==1].groupby('Tenure_binned').size() / len(df[df['Churn']==1])
print(tenure_churn)

print("\n" + "="*40)
print("SERVICE COMBINATION CHURN RISK")
service_risk = df.groupby('internet_tech_2_0')['Churn'].mean()
# Check stats for fiber optic internet (2) + no tech support (0)
print(f"Churn rate for fiber optic internet + No tech support: {service_risk[1]:.1%}")

print("\n" + "="*40)
print("HIGH-RISK CUSTOMER PROFILE")
high_risk = df[
    (df['Contract']==0) & 
    (df['Tenure_binned']<3) & 
    (df['internet_tech_2_0']==1)
]
print(f"Churn rate for Month-to-Month contract, Tenure less than 12 and using Fiber Optic with no Tech Support is: {high_risk['Churn'].mean():.1%}")


# =====================
# 5. VISUAL COMPARISON
# =====================
print("Contract Type Churn Risk\n")
plt.figure(figsize=(10, 6))
contract_churn = df.groupby('Contract')['Churn'].mean().sort_values(ascending=False)
x_label = ['Month-to-Month', 'One Year', 'Two Year']
ax = sns.barplot(x=x_label, y=contract_churn.values, palette=['#ff6b6b', '#feca57', '#1dd1a1'])

plt.title('Churn Rate by Contract Type', fontsize=16, pad=20)
plt.xlabel('Contract Type', fontsize=12)
plt.ylabel('Churn Rate', fontsize=12)
plt.ylim(0, 0.5)
for i, v in enumerate(contract_churn.values):
    ax.text(i, v+0.02, f"{v:.1%}", ha='center', fontsize=12)
    
plt.axhline(y=df['Churn'].mean(), color='red', linestyle='--', label='Overall Churn')
plt.legend()
plt.tight_layout()
plt.show()



print("Tenure Binned Churn Distribution\n")
plt.figure(figsize=(12, 6))
tenure_churn = df[df['Churn']==1].groupby('Tenure_binned').size() / len(df[df['Churn']==1])
# Convert bins to readable labels
tenure_bin_edges = tenure_binner.bin_edges_[0]
bin_labels = tenure_bin_edges[:-1].astype(str) + '–' + tenure_bin_edges[1:].astype(str)
plt.bar(bin_labels, tenure_churn, color='#3498db')
plt.title('Churn Distribution by Tenure', fontsize=16, pad=20)
plt.xlabel('Tenure Range', fontsize=12)
plt.ylabel('Percentage of Total Churn', fontsize=12)
plt.xticks(rotation=45, ha='right')
# Highlight critical period
plt.axvspan(-0.5, 2.5, color='red', alpha=0.1, label='Critical Risk Period')
plt.text(1, max(tenure_churn)*0.9, f"{bin_0_to_2_churners_pct:.1f}% of churn\noccurs here", ha='center', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()


print("Service(Fiber Optic and no Tech Support) Churn Distribution\n")
service_labels = ['Other Combinations', 'Fiber Optic + No Tech Support']
plt.figure(figsize=(8, 6))
service_data = df.groupby('internet_tech_2_0')['Churn'].mean()
ax = sns.barplot(x=service_labels, 
                y=service_data.values, 
                palette=['#feca57', '#ff6b6b'])
plt.title('Churn Rate by Service Combination', fontsize=16, pad=20)
plt.xlabel('')
plt.ylabel('Churn Rate', fontsize=12)
plt.ylim(0, 0.7)
# Add difference arrow
ax.annotate('', xy=(1, 0.58), xytext=(0, 0.2),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(0.5, 0.4, f"{service_data[1]/service_data[0]:.1f}x higher risk", 
        ha='center', fontsize=12)
plt.tight_layout()
plt.show()


print("High-Risk Customer Profile\n")   
# Radar Chart for High-Risk Customer Profile
plt.figure(figsize=(10, 8))
ax = plt.subplot(111, polar=True)
# Data for radar chart
categories = ['Contract Risk (Month-to-Month)', 'Tenure Risk (<12 months)', 'Service Risk (fiber optic + no tech support)']
values = [
    df[df['Contract']==0]['Churn'].mean() / df['Churn'].mean(),
    df[df['Tenure_binned']<3]['Churn'].mean() / df['Churn'].mean(),
    df[df['internet_tech_2_0']==1]['Churn'].mean() / df['Churn'].mean()
]
values += values[:1]
# Plot
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]
plt.polar(angles, values, color='red', linewidth=3)
plt.fill(angles, values, color='red', alpha=0.1)
# Customize
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], categories, fontsize=12)
plt.yticks([1, 2, 3], ["1x Avg", "2x Avg", "3x Avg"], fontsize=10)
plt.title('High-Risk Customer Profile', fontsize=16, pad=20)
# Add stats
plt.figtext(0.5, 0.15, 
           f"These customers are {high_risk['Churn'].mean()/df['Churn'].mean():.1f}x more likely to churn\n"
           f"Represent {len(high_risk)/len(df):.1%} of customers but drive {len(high_risk[high_risk['Churn']==1])/len(df[df['Churn']==1]):.1%} of churn",
           ha='center', fontsize=12)
plt.tight_layout()
plt.show()


# Save all models and preprocessors
joblib.dump(rf, 'random_forest_model.pkl')
joblib.dump(gb, 'gradient_boosting_model.pkl')
joblib.dump(xgb_model, 'xgboost_model.pkl')
joblib.dump(monthly_binner, 'monthly_binner.pkl') 
joblib.dump(total_binner, 'total_binner.pkl')      
joblib.dump(tenure_binner, 'tenure_binner.pkl')    
joblib.dump(X.columns, 'feature_columns.pkl')
joblib.dump(xgb_threshold, 'xgb_threshold.pkl')

print("Models saved successfully!")
