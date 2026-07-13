#PART 2 - SUPERVISED MACHINE LEARNING

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score,recall_score,f1_score,roc_curve,roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
import joblib

print("=" * 80)
print("LOADING CLEAN DATASET")
print("=" * 80)
df = pd.read_csv("cleaned_data.csv")
print(df.head())
print("\nDataset Shape")
print(df.shape)
print("\nData Types")
print(df.dtypes)

# DEFINE TARGET VARIABLES
print("\n" + "=" * 80)
print("TARGET VARIABLES")
print("=" * 80)

# Regression Target
y_reg = df["MonthlyCharges"]
# Classification Target
y_clf = df["Churn"].map({"No": 0,"Yes": 1})
print("\nRegression Target")
print("MonthlyCharges")
print("\nClassification Target")
print(y_clf.value_counts())

# FEATURE MATRIX
print("\n" + "=" * 80)
print("FEATURE MATRIX")
print("=" * 80)
X = df.drop(columns=["MonthlyCharges","Churn"])
print(X.head())

# ENCODE CATEGORICAL COLUMNS
print("\n" + "=" * 80)
print("ENCODING CATEGORICAL FEATURES")
print("=" * 80)
categorical_columns = X.select_dtypes(include=["object","category"]).columns
print("\nCategorical Columns")
print(list(categorical_columns))

# ORDINAL ENCODING
contract_mapping = {"Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}
if "Contract" in X.columns:
    X["Contract"] = X["Contract"].map(contract_mapping)
    print("\nContract column encoded using ordinal mapping.")

# ONE HOT ENCODING
remaining_categorical = X.select_dtypes(include=["object","category"]).columns
X = pd.get_dummies(X,columns=remaining_categorical,drop_first=True)
print("\nEncoded Feature Shape")
print(X.shape)

# TRAIN TEST SPLIT
print("\n" + "=" * 80)
print("TRAIN TEST SPLIT")
print("=" * 80)
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X,y_reg,test_size=0.2,random_state=42)
_, _, y_train_clf, y_test_clf = train_test_split(X,y_clf,test_size=0.2,random_state=42)

print("\nTraining Feature Shape")
print(X_train.shape)
print("\nTesting Feature Shape")
print(X_test.shape)
print("\nRegression Training Labels")
print(y_train_reg.shape)
print("\nRegression Testing Labels")
print(y_test_reg.shape)
print("\nClassification Training Labels")
print(y_train_clf.shape)
print("\nClassification Testing Labels")
print(y_test_clf.shape)


# FEATURE SCALING
print("\n" + "=" * 80)
print("STANDARD SCALING")
print("=" * 80)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaled Training Shape")
print(X_train_scaled.shape)
print("\nScaled Testing Shape")
print(X_test_scaled.shape)

# LINEAR REGRESSION
print("\n" + "=" * 80)
print("LINEAR REGRESSION")
print("=" * 80)
linear_model = LinearRegression()
linear_model.fit(X_train_scaled,y_train_reg)
y_pred_linear = linear_model.predict(X_test_scaled)

mse_linear = mean_squared_error(y_test_reg,y_pred_linear)
r2_linear = r2_score(y_test_reg,y_pred_linear)

print("\nLinear Regression Performance")
print(f"MSE: {mse_linear:.4f}")
print(f"R²: {r2_linear:.4f}")

# FEATURE COEFFICIENTS
print("\n" + "=" * 80)
print("FEATURE COEFFICIENTS")
print("=" * 80)
coefficients = pd.DataFrame({"Feature": X.columns,
    "Coefficient": linear_model.coef_
})
coefficients["Absolute"] = coefficients["Coefficient"].abs()
coefficients = coefficients.sort_values(by="Absolute",ascending=False)

print(coefficients)
print("\nTop 3 Important Features")
print(coefficients.head(3)[["Feature","Coefficient"]])

# RIDGE REGRESSION
print("\n" + "=" * 80)
print("RIDGE REGRESSION")
print("=" * 80)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled,y_train_reg)
y_pred_ridge = ridge_model.predict(X_test_scaled)
mse_ridge = mean_squared_error(y_test_reg,y_pred_ridge)
r2_ridge = r2_score(y_test_reg,y_pred_ridge)

print("\nRidge Regression Performance")
print(f"MSE: {mse_ridge:.4f}")
print(f"R²: {r2_ridge:.4f}")

# MODEL COMPARISON
print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)
comparison = pd.DataFrame({"Model":[
        "Linear Regression",
        "Ridge Regression"
    ],
    "MSE":[
        mse_linear,
        mse_ridge
    ],
    "R2":[
        r2_linear,
        r2_ridge
    ]
})

print(comparison)

# LOGISTIC REGRESSION
print("\n" + "=" * 80)
print("LOGISTIC REGRESSION")
print("=" * 80)

# CHECK CLASS BALANCE
print("\nClass Distribution Before Training")
print(y_train_clf.value_counts())
class_ratio = y_train_clf.value_counts(normalize=True)
print("\nClass Percentage")
print(class_ratio)

# HANDLE CLASS IMBALANCE
logistic_model = LogisticRegression(class_weight="balanced",max_iter=1000,random_state=42)
logistic_model.fit(X_train_scaled,y_train_clf)

# PREDICTIONS
y_pred = logistic_model.predict(X_test_scaled)
y_prob = logistic_model.predict_proba(X_test_scaled)[:,1]

# CONFUSION MATRIX
print("\nConfusion Matrix")
cm = confusion_matrix(y_test_clf,y_pred)
print(cm)

# METRICS
accuracy = accuracy_score(y_test_clf,y_pred)
precision = precision_score(y_test_clf,y_pred)
recall = recall_score(y_test_clf,y_pred)
f1 = f1_score(y_test_clf,y_pred)

print("\nAccuracy :", round(accuracy,4))
print("Precision:", round(precision,4))
print("Recall   :", round(recall,4))
print("F1 Score :", round(f1,4))
print("\nClassification Report")
print(classification_report(y_test_clf,y_pred))

# ROC CURVE
fpr, tpr, thresholds = roc_curve(y_test_clf,y_prob)
auc = roc_auc_score(y_test_clf,y_prob)
print("\nROC AUC :", round(auc,4))
plt.figure(figsize=(8,6))
plt.plot(fpr,tpr,label=f"AUC = {auc:.3f}")
plt.plot([0,1],[0,1],linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png",dpi=300)
plt.close()
print("\nROC Curve Saved.")

# DECISION THRESHOLD SENSITIVITY
print("\n" + "=" * 80)
print("DECISION THRESHOLD SENSITIVITY")
print("=" * 80)
thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
results = []
for threshold in thresholds:
    y_pred_threshold = (y_prob >= threshold).astype(int)
    precision_score(y_test_clf, y_pred_threshold)
    recall = recall_score(y_test_clf,y_pred)
    f1 = f1_score(y_test_clf,y_pred)

    results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })
threshold_results = pd.DataFrame(results)
print("\nThreshold Comparison")
print(threshold_results)
best_row = threshold_results.loc[threshold_results["F1"].idxmax()]
print("\nBest Threshold Based on F1 Score")
print(best_row)

# SECOND MODEL WITH STRONGER REGULARIZATION
logistic_model_c001 = LogisticRegression(C=0.01,class_weight="balanced",max_iter=1000,random_state=42)
logistic_model_c001.fit(X_train_scaled, y_train_clf)
# PREDICTIONS
y_pred_c001 = logistic_model_c001.predict(X_test_scaled)
y_prob_c001 = logistic_model_c001.predict_proba(X_test_scaled)[:, 1]

# METRICS
precision_c001 = precision_score(y_test_clf, y_pred_c001)
recall_c001 = recall_score(y_test_clf, y_pred_c001)
auc_c001 = roc_auc_score(y_test_clf, y_prob_c001)

print("\nLogistic Regression (C=0.01)")
print("Precision:", round(precision_c001, 4))
print("Recall:", round(recall_c001, 4))
print("AUC:", round(auc_c001, 4))

baseline_probability = y_prob
regularized_probability = y_prob_c001
#Bootstraps
n_bootstraps = 500
rng = np.random.RandomState(42)
auc_differences = []
for i in range(n_bootstraps):
    indices = rng.choice(len(y_test_clf),size=len(y_test_clf),replace=True)
    # Bootstrap sample
    y_true_boot = y_test_clf.iloc[indices]
    baseline_boot = baseline_probability[indices]
    regularized_boot = regularized_probability[indices]
    # AUC requires both classes to be present
    if len(np.unique(y_true_boot)) < 2:
        continue
    # Calculate AUCs
    auc_baseline = roc_auc_score(y_true_boot, baseline_boot)
    auc_regularized = roc_auc_score(y_true_boot, regularized_boot)

    # Store difference
    auc_differences.append(auc_baseline - auc_regularized)
# Number of successful bootstrap samples
print(f"\nValid Bootstrap Samples: {len(auc_differences)}")
# Safety check
if len(auc_differences) == 0:
    print("Error: No valid bootstrap samples were generated.")
else:
    auc_differences = np.array(auc_differences)
    mean_difference = np.mean(auc_differences)
    lower_ci = np.percentile(auc_differences, 2.5)
    upper_ci = np.percentile(auc_differences, 97.5)
    print("\n" + "=" * 80)
    print("BOOTSTRAP CONFIDENCE INTERVAL FOR AUC DIFFERENCE")
    print("=" * 80)
    print(f"\nMean Difference: {mean_difference:.4f}")
    print("\n95% Confidence Interval")
    print(f"Lower: {lower_ci:.4f}")
    print(f"Upper: {upper_ci:.4f}")
   # Interpretation required by assignment
    print("\nInterpretation")
    if lower_ci > 0:
        print("The baseline model consistently performs better.")
        print("The entire 95% confidence interval is above zero.")
    else:
        print("The confidence interval includes zero.")
        print("The difference between the two models is not statistically reliable.")

