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

# print("=" * 80)
# print("LOADING CLEAN DATASET")
# print("=" * 80)
# df = pd.read_csv("cleaned_data.csv")
# print(df.head())
# print("\nDataset Shape")
# print(df.shape)
# print("\nData Types")
# print(df.dtypes)
#
# # DEFINE TARGET VARIABLES
# print("\n" + "=" * 80)
# print("TARGET VARIABLES")
# print("=" * 80)
#
# # Regression Target
# y_reg = df["MonthlyCharges"]
# # Classification Target
# y_clf = df["Churn"].map({"No": 0,"Yes": 1})
# print("\nRegression Target")
# print("MonthlyCharges")
# print("\nClassification Target")
# print(y_clf.value_counts())
#
# # FEATURE MATRIX
# print("\n" + "=" * 80)
# print("FEATURE MATRIX")
# print("=" * 80)
# X = df.drop(columns=["MonthlyCharges","Churn"])
# print(X.head())
#
# # ENCODE CATEGORICAL COLUMNS
# print("\n" + "=" * 80)
# print("ENCODING CATEGORICAL FEATURES")
# print("=" * 80)
# categorical_columns = X.select_dtypes(include=["object","category"]).columns
# print("\nCategorical Columns")
# print(list(categorical_columns))
#
# # ORDINAL ENCODING
# contract_mapping = {"Month-to-month": 0,
#     "One year": 1,
#     "Two year": 2
# }
# if "Contract" in X.columns:
#     X["Contract"] = X["Contract"].map(contract_mapping)
#     print("\nContract column encoded using ordinal mapping.")
#
# # ONE HOT ENCODING
# remaining_categorical = X.select_dtypes(include=["object","category"]).columns
# X = pd.get_dummies(X,columns=remaining_categorical,drop_first=True)
# print("\nEncoded Feature Shape")
# print(X.shape)
#
# # TRAIN TEST SPLIT
# print("\n" + "=" * 80)
# print("TRAIN TEST SPLIT")
# print("=" * 80)
# X_train, X_test, y_train_reg, y_test_reg = train_test_split(X,y_reg,test_size=0.2,random_state=42)
# _, _, y_train_clf, y_test_clf = train_test_split(X,y_clf,test_size=0.2,random_state=42)
#
# print("\nTraining Feature Shape")
# print(X_train.shape)
# print("\nTesting Feature Shape")
# print(X_test.shape)
# print("\nRegression Training Labels")
# print(y_train_reg.shape)
# print("\nRegression Testing Labels")
# print(y_test_reg.shape)
# print("\nClassification Training Labels")
# print(y_train_clf.shape)
# print("\nClassification Testing Labels")
# print(y_test_clf.shape)
#
#
# # FEATURE SCALING
# print("\n" + "=" * 80)
# print("STANDARD SCALING")
# print("=" * 80)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# print("\nScaled Training Shape")
# print(X_train_scaled.shape)
# print("\nScaled Testing Shape")
# print(X_test_scaled.shape)
#
# # LINEAR REGRESSION
# print("\n" + "=" * 80)
# print("LINEAR REGRESSION")
# print("=" * 80)
# linear_model = LinearRegression()
# linear_model.fit(X_train_scaled,y_train_reg)
# y_pred_linear = linear_model.predict(X_test_scaled)
#
# mse_linear = mean_squared_error(y_test_reg,y_pred_linear)
# r2_linear = r2_score(y_test_reg,y_pred_linear)
#
# print("\nLinear Regression Performance")
# print(f"MSE: {mse_linear:.4f}")
# print(f"R²: {r2_linear:.4f}")
#
# # FEATURE COEFFICIENTS
# print("\n" + "=" * 80)
# print("FEATURE COEFFICIENTS")
# print("=" * 80)
# coefficients = pd.DataFrame({"Feature": X.columns,
#     "Coefficient": linear_model.coef_
# })
# coefficients["Absolute"] = coefficients["Coefficient"].abs()
# coefficients = coefficients.sort_values(by="Absolute",ascending=False)
#
# print(coefficients)
# print("\nTop 3 Important Features")
# print(coefficients.head(3)[["Feature","Coefficient"]])
#
# # RIDGE REGRESSION
# print("\n" + "=" * 80)
# print("RIDGE REGRESSION")
# print("=" * 80)
# ridge_model = Ridge(alpha=1.0)
# ridge_model.fit(X_train_scaled,y_train_reg)
# y_pred_ridge = ridge_model.predict(X_test_scaled)
# mse_ridge = mean_squared_error(y_test_reg,y_pred_ridge)
# r2_ridge = r2_score(y_test_reg,y_pred_ridge)
#
# print("\nRidge Regression Performance")
# print(f"MSE: {mse_ridge:.4f}")
# print(f"R²: {r2_ridge:.4f}")
#
# # MODEL COMPARISON
# print("\n" + "=" * 80)
# print("MODEL COMPARISON")
# print("=" * 80)
# comparison = pd.DataFrame({"Model":[
#         "Linear Regression",
#         "Ridge Regression"
#     ],
#     "MSE":[
#         mse_linear,
#         mse_ridge
#     ],
#     "R2":[
#         r2_linear,
#         r2_ridge
#     ]
# })
#
# print(comparison)
#
# # LOGISTIC REGRESSION
# print("\n" + "=" * 80)
# print("LOGISTIC REGRESSION")
# print("=" * 80)
#
# # CHECK CLASS BALANCE
# print("\nClass Distribution Before Training")
# print(y_train_clf.value_counts())
# class_ratio = y_train_clf.value_counts(normalize=True)
# print("\nClass Percentage")
# print(class_ratio)
#
# # HANDLE CLASS IMBALANCE
# logistic_model = LogisticRegression(class_weight="balanced",max_iter=1000,random_state=42)
# logistic_model.fit(X_train_scaled,y_train_clf)
#
# # PREDICTIONS
# y_pred = logistic_model.predict(X_test_scaled)
# y_prob = logistic_model.predict_proba(X_test_scaled)[:,1]
#
# # CONFUSION MATRIX
# print("\nConfusion Matrix")
# cm = confusion_matrix(y_test_clf,y_pred)
# print(cm)
#
# # METRICS
# accuracy = accuracy_score(y_test_clf,y_pred)
# precision = precision_score(y_test_clf,y_pred)
# recall = recall_score(y_test_clf,y_pred)
# f1 = f1_score(y_test_clf,y_pred)
#
# print("\nAccuracy :", round(accuracy,4))
# print("Precision:", round(precision,4))
# print("Recall   :", round(recall,4))
# print("F1 Score :", round(f1,4))
# print("\nClassification Report")
# print(classification_report(y_test_clf,y_pred))
#
# # ROC CURVE
# fpr, tpr, thresholds = roc_curve(y_test_clf,y_prob)
# auc = roc_auc_score(y_test_clf,y_prob)
# print("\nROC AUC :", round(auc,4))
# plt.figure(figsize=(8,6))
# plt.plot(fpr,tpr,label=f"AUC = {auc:.3f}")
# plt.plot([0,1],[0,1],linestyle="--")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend()
# plt.tight_layout()
# plt.savefig("roc_curve.png",dpi=300)
# plt.close()
# print("\nROC Curve Saved.")
#
# # DECISION THRESHOLD SENSITIVITY
# print("\n" + "=" * 80)
# print("DECISION THRESHOLD SENSITIVITY")
# print("=" * 80)
# thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
# results = []
# for threshold in thresholds:
#     y_pred_threshold = (y_prob >= threshold).astype(int)
#     precision_score(y_test_clf, y_pred_threshold)
#     recall = recall_score(y_test_clf,y_pred)
#     f1 = f1_score(y_test_clf,y_pred)
#
#     results.append({
#         "Threshold": threshold,
#         "Precision": precision,
#         "Recall": recall,
#         "F1": f1
#     })
# threshold_results = pd.DataFrame(results)
# print("\nThreshold Comparison")
# print(threshold_results)
# best_row = threshold_results.loc[threshold_results["F1"].idxmax()]
# print("\nBest Threshold Based on F1 Score")
# print(best_row)
#
# # SECOND MODEL WITH STRONGER REGULARIZATION
# logistic_model_c001 = LogisticRegression(C=0.01,class_weight="balanced",max_iter=1000,random_state=42)
# logistic_model_c001.fit(X_train_scaled, y_train_clf)
# # PREDICTIONS
# y_pred_c001 = logistic_model_c001.predict(X_test_scaled)
# y_prob_c001 = logistic_model_c001.predict_proba(X_test_scaled)[:, 1]
#
# # METRICS
# precision_c001 = precision_score(y_test_clf, y_pred_c001)
# recall_c001 = recall_score(y_test_clf, y_pred_c001)
# auc_c001 = roc_auc_score(y_test_clf, y_prob_c001)
#
# print("\nLogistic Regression (C=0.01)")
# print("Precision:", round(precision_c001, 4))
# print("Recall:", round(recall_c001, 4))
# print("AUC:", round(auc_c001, 4))
#
# baseline_probability = y_prob
# regularized_probability = y_prob_c001
# #Bootstraps
# n_bootstraps = 500
# rng = np.random.RandomState(42)
# auc_differences = []
# for i in range(n_bootstraps):
#     indices = rng.choice(len(y_test_clf),size=len(y_test_clf),replace=True)
#     # Bootstrap sample
#     y_true_boot = y_test_clf.iloc[indices]
#     baseline_boot = baseline_probability[indices]
#     regularized_boot = regularized_probability[indices]
#     # AUC requires both classes to be present
#     if len(np.unique(y_true_boot)) < 2:
#         continue
#     # Calculate AUCs
#     auc_baseline = roc_auc_score(y_true_boot, baseline_boot)
#     auc_regularized = roc_auc_score(y_true_boot, regularized_boot)
#
#     # Store difference
#     auc_differences.append(auc_baseline - auc_regularized)
# # Number of successful bootstrap samples
# print(f"\nValid Bootstrap Samples: {len(auc_differences)}")
# # Safety check
# if len(auc_differences) == 0:
#     print("Error: No valid bootstrap samples were generated.")
# else:
#     auc_differences = np.array(auc_differences)
#     mean_difference = np.mean(auc_differences)
#     lower_ci = np.percentile(auc_differences, 2.5)
#     upper_ci = np.percentile(auc_differences, 97.5)
#     print("\n" + "=" * 80)
#     print("BOOTSTRAP CONFIDENCE INTERVAL FOR AUC DIFFERENCE")
#     print("=" * 80)
#     print(f"\nMean Difference: {mean_difference:.4f}")
#     print("\n95% Confidence Interval")
#     print(f"Lower: {lower_ci:.4f}")
#     print(f"Upper: {upper_ci:.4f}")
#    # Interpretation required by assignment
#     print("\nInterpretation")
#     if lower_ci > 0:
#         print("The baseline model consistently performs better.")
#         print("The entire 95% confidence interval is above zero.")
#     else:
#         print("The confidence interval includes zero.")
#         print("The difference between the two models is not statistically reliable.")

# ==============================================================================
# PART 3A - DECISION TREE MODELS
# ==============================================================================

# DECISION TREE - BASELINE
print("\n" + "=" * 80)
print("BASELINE DECISION TREE")
print("=" * 80)
baseline_tree = DecisionTreeClassifier(random_state=42)
baseline_tree.fit(X_train_scaled,y_train_clf)
train_pred = baseline_tree.predict(X_train_scaled)
test_pred = baseline_tree.predict(X_test_scaled)
train_accuracy = accuracy_score(y_train_clf,train_pred)
test_accuracy = accuracy_score(y_test_clf,test_pred)

print("\nTraining Accuracy :", round(train_accuracy, 4))
print("Testing Accuracy  :", round(test_accuracy, 4))
gap = train_accuracy - test_accuracy
print("Train-Test Gap    :", round(gap, 4))

# CONTROLLED DECISION TREE
print("\n" + "=" * 80)
print("CONTROLLED DECISION TREE")
print("=" * 80)
controlled_tree = DecisionTreeClassifier(max_depth=5,min_samples_split=20,random_state=42)
controlled_tree.fit(X_train_scaled,y_train_clf)
train_pred_control = controlled_tree.predict(X_train_scaled)
test_pred_control = controlled_tree.predict(X_test_scaled)
train_accuracy_control = accuracy_score(y_train_clf,train_pred_control)
test_accuracy_control = accuracy_score(y_test_clf,test_pred_control)

print("\nTraining Accuracy :", round(train_accuracy_control, 4))
print("Testing Accuracy  :", round(test_accuracy_control, 4))
gap_control = train_accuracy_control - test_accuracy_control
print("Train-Test Gap    :", round(gap_control, 4))

# COMPARISON
print("\n" + "=" * 80)
print("BASELINE VS CONTROLLED TREE")
print("=" * 80)
comparison = pd.DataFrame({
    "Model": [
        "Baseline Tree",
        "Controlled Tree"
    ],
    "Training Accuracy": [
        train_accuracy,
        train_accuracy_control
    ],
    "Testing Accuracy": [
        test_accuracy,
        test_accuracy_control
    ],
    "Gap": [
        gap,
        gap_control
    ]
})
print(comparison)

# GINI VS ENTROPY
print("\n" + "=" * 80)
print("GINI VS ENTROPY")
print("=" * 80)
gini_tree = DecisionTreeClassifier(criterion="gini",max_depth=5,random_state=42)
entropy_tree = DecisionTreeClassifier(criterion="entropy",max_depth=5,random_state=42)
gini_tree.fit(X_train_scaled,y_train_clf)
entropy_tree.fit(X_train_scaled,y_train_clf)
gini_pred = gini_tree.predict(X_test_scaled)
entropy_pred = entropy_tree.predict(X_test_scaled)
gini_accuracy = accuracy_score(y_test_clf,gini_pred)
entropy_accuracy = accuracy_score(y_test_clf,entropy_pred)

print("\nGini Test Accuracy    :", round(gini_accuracy, 4))
print("Entropy Test Accuracy :", round(entropy_accuracy, 4))

# SUMMARY TABLE
print("\n" + "=" * 80)
print("DECISION TREE SUMMARY")
print("=" * 80)
summary = pd.DataFrame({
    "Model": [
        "Baseline Decision Tree",
        "Controlled Decision Tree",
        "Gini Decision Tree",
        "Entropy Decision Tree"
    ],
    "Train Accuracy": [
        train_accuracy,
        train_accuracy_control,
        "-",
        "-"
    ],
    "Test Accuracy": [
        test_accuracy,
        test_accuracy_control,
        gini_accuracy,
        entropy_accuracy
    ]
})

print(summary)

# RANDOM FOREST CLASSIFIER
print("\n" + "=" * 80)
print("RANDOM FOREST CLASSIFIER")
print("=" * 80)
rf_model = RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42)
rf_model.fit(X_train_scaled,y_train_clf)

# Predictions
rf_train_pred = rf_model.predict(X_train_scaled)
rf_test_pred = rf_model.predict(X_test_scaled)
rf_train_prob = rf_model.predict_proba(X_train_scaled)[:, 1]
rf_test_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

# Metrics
rf_train_accuracy = accuracy_score(y_train_clf,rf_train_pred)
rf_test_accuracy = accuracy_score(y_test_clf,rf_test_pred)
rf_auc = roc_auc_score(y_test_clf,rf_test_prob)

print("\nTraining Accuracy :", round(rf_train_accuracy,4))
print("Testing Accuracy  :", round(rf_test_accuracy,4))
print("ROC-AUC Score     :", round(rf_auc,4))

# FEATURE IMPORTANCE
print("\n" + "=" * 80)
print("TOP 5 FEATURE IMPORTANCE")
print("=" * 80)
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})
importance = importance.sort_values(by="Importance",ascending=False)
top5 = importance.head(5)
print(top5)

# GRADIENT BOOSTING CLASSIFIER
print("\n" + "=" * 80)
print("GRADIENT BOOSTING CLASSIFIER")
print("=" * 80)
gb_model = GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=42)
gb_model.fit(X_train_scaled,y_train_clf)
# Predictions
gb_train_pred = gb_model.predict(X_train_scaled)
gb_test_pred = gb_model.predict(X_test_scaled)
gb_test_prob = gb_model.predict_proba(X_test_scaled)[:,1]

# Metrics
gb_train_accuracy = accuracy_score(y_train_clf,gb_train_pred)
gb_test_accuracy = accuracy_score(y_test_clf,gb_test_pred)
gb_auc = roc_auc_score(y_test_clf,gb_test_prob)

print("\nTraining Accuracy :", round(gb_train_accuracy,4))
print("Testing Accuracy  :", round(gb_test_accuracy,4))
print("ROC-AUC Score     :", round(gb_auc,4))

# FEATURE ABLATION STUDY
print("\n" + "=" * 80)
print("FEATURE ABLATION STUDY")
print("=" * 80)

lowest5 = importance.tail(5)
print("\nLowest 5 Important Features")
print(lowest5)
remove_features = lowest5["Feature"].tolist()
X_train_reduced = X_train.drop(columns=remove_features)
X_test_reduced = X_test.drop(columns=remove_features)

# Scale reduced dataset
scaler_reduced = StandardScaler()
X_train_reduced_scaled = scaler_reduced.fit_transform(X_train_reduced)
X_test_reduced_scaled = scaler_reduced.transform(X_test_reduced)
rf_reduced = RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42)
rf_reduced.fit(X_train_reduced_scaled,y_train_clf)
reduced_prob = rf_reduced.predict_proba(X_test_reduced_scaled)[:,1]
reduced_auc = roc_auc_score(y_test_clf,reduced_prob
)
print("\nFull Model ROC-AUC    :", round(rf_auc,4))
print("Reduced Model ROC-AUC :", round(reduced_auc,4))
print("\nAUC Difference :", round(rf_auc - reduced_auc,4))

# SUMMARY
print("\n" + "=" * 80)
print("ENSEMBLE MODEL SUMMARY")
print("=" * 80)
summary = pd.DataFrame({
    "Model":[
        "Random Forest",
        "Gradient Boosting"
    ],
    "Training Accuracy":[
        rf_train_accuracy,
        gb_train_accuracy
    ],
    "Testing Accuracy":[
        rf_test_accuracy,
        gb_test_accuracy
    ],
    "ROC-AUC":[
        rf_auc,
        gb_auc
    ]
})

print(summary)

# FOLD CROSS VALIDATION
print("\n" + "=" * 80)
print("5-FOLD STRATIFIED CROSS VALIDATION")
print("=" * 80)
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

# Logistic Regression
logistic_pipeline = make_pipeline(StandardScaler(),LogisticRegression(class_weight="balanced",max_iter=1000,random_state=42))
logistic_scores = cross_val_score(logistic_pipeline,X,y_clf,cv=cv,scoring="roc_auc",n_jobs=-1)

# Controlled Decision Tree
tree_pipeline = make_pipeline(StandardScaler(),DecisionTreeClassifier(max_depth=5,min_samples_split=20,random_state=42))
tree_scores = cross_val_score(tree_pipeline,X,y_clf,cv=cv,scoring="roc_auc",n_jobs=-1)

# Random Forest
rf_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42))
rf_scores = cross_val_score(rf_pipeline,X,y_clf,cv=cv,scoring="roc_auc",n_jobs=-1)

# Gradient Boosting
gb_pipeline = make_pipeline(StandardScaler(),GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=42))
gb_scores = cross_val_score(gb_pipeline,X,y_clf,cv=cv,scoring="roc_auc",n_jobs=-1)

# SUMMARY
cv_results = pd.DataFrame({
    "Model":[
        "Logistic Regression",
        "Controlled Decision Tree",
        "Random Forest",
        "Gradient Boosting"
    ],
    "Mean AUC":[
        logistic_scores.mean(),
        tree_scores.mean(),
        rf_scores.mean(),
        gb_scores.mean()
    ],
    "Std AUC":[
        logistic_scores.std(),
        tree_scores.std(),
        rf_scores.std(),
        gb_scores.std()
    ]
})
print("\nCross Validation Results")
print(cv_results)
print("\nBest Model Based on Mean AUC")
best_model = cv_results.sort_values(by="Mean AUC",ascending=False).iloc[0]
print(best_model)
print("\nCross Validation Completed Successfully.")

# PIPELINE + GRID SEARCH CV
print("\n" + "=" * 80)
print("PIPELINE + GRID SEARCH CV")
print("=" * 80)
# BUILD PIPELINE
pipeline = make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),RandomForestClassifier(random_state=42))
print("\nPipeline Created Successfully.")

# PARAMETER GRID
param_grid = {
    "randomforestclassifier__n_estimators": [
        50,
        100,
        200
    ],
    "randomforestclassifier__max_depth": [
        5,
        10,
        None
    ],
    "randomforestclassifier__min_samples_leaf": [
        1,
        5
    ]
}
print("\nParameter Grid")
print(param_grid)

# STRATIFIED K-FOLD
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

# GRID SEARCH
grid_search = GridSearchCV(estimator=pipeline,param_grid=param_grid,cv=cv,scoring="roc_auc",n_jobs=-1,verbose=1)
print("\nRunning Grid Search...")
grid_search.fit(X_train,y_train_clf)

# RESULTS
print("\n" + "=" * 80)
print("GRID SEARCH RESULTS")
print("=" * 80)
print("\nBest Parameters")
print(grid_search.best_params_)
print("\nBest Cross Validation AUC")
print(round(grid_search.best_score_,4))
best_pipeline = grid_search.best_estimator_

# TOTAL MODELS EVALUATED
total_models = (3 *3 *2)
total_fits = total_models * 5
print("\nTotal Parameter Combinations :", total_models)
print("Total Models Trained :", total_fits)

# TEST SET PERFORMANCE
best_probability = best_pipeline.predict_proba(X_test)[:,1]
best_prediction = best_pipeline.predict(X_test)
best_auc = roc_auc_score(y_test_clf,best_probability)
best_accuracy = accuracy_score(y_test_clf,best_prediction)
print("\nBest Pipeline Test Accuracy :", round(best_accuracy,4))
print("Best Pipeline Test ROC-AUC  :", round(best_auc,4))
print("\nGrid Search Completed Successfully.")

# MANUAL LEARNING CURVE
print("\n" + "=" * 80)
print("MANUAL LEARNING CURVE")
print("=" * 80)
fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
learning_results = []
for fraction in fractions:
    size = int(fraction * len(X_train))
    X_subset = X_train.iloc[:size]
    y_subset = y_train_clf.iloc[:size]
    # Train pipeline on subset
    best_pipeline.fit(X_subset, y_subset)
    # Training AUC
    train_prob = best_pipeline.predict_proba(X_subset)[:, 1]
    train_auc = roc_auc_score(y_subset, train_prob)
    # Test AUC
    test_prob = best_pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test_clf, test_prob)
    learning_results.append([fraction,train_auc,test_auc])
learning_curve = pd.DataFrame(learning_results,columns=["Training Fraction","Training AUC","Test AUC"])
print("\nLearning Curve")
print(learning_curve)

# SAVE BEST MODEL
print("\n" + "=" * 80)
print("MODEL SERIALIZATION")
print("=" * 80)
joblib.dump(best_pipeline,"best_model.pkl")
print("Model saved as best_model.pkl")

# LOAD MODEL AND PREDICT
print("\n" + "=" * 80)
print("LOADING SAVED MODEL")
print("=" * 80)
loaded_model = joblib.load("best_model.pkl")
print("Model Loaded Successfully")
# Create two sample rows
sample_rows = X_test.iloc[:2]
predictions = loaded_model.predict(sample_rows)
probabilities = loaded_model.predict_proba(sample_rows)
print("\nPredictions")
print(predictions)

print("\nPrediction Probabilities")
print(probabilities)
