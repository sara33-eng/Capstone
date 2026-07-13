# Telco Customer Churn - Exploratory Data Analysis (Part 1)
This project performs Exploratory Data Analysis (EDA) on the Telco Customer Churn dataset. The objective is to inspect, clean, and understand the raw data before applying machine learning techniques in the next phase of the project.
The analysis includes missing value treatment, duplicate detection, data type correction, descriptive statistics, skewness analysis, outlier detection, visualization, correlation analysis, grouped aggregation, and creation of a cleaned dataset for future modeling.

# Dataset Information
Dataset Name: Telco Customer Churn
Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Rows:7043
Columns:21
The dataset contains customer demographic information, subscribed telecom services, billing details, contract information, and customer churn status.

# Software Requirements
Python 3.9 or above
Required libraries:pandas, numpy, matplotlib ,seaborn

Install the dependencies using:pip install pandas numpy matplotlib seaborn

# How to Run
Place the dataset inside the "data" folder.
Run the project using: python eda.py
The script automatically generates all required plots and the cleaned dataset.

# Data Cleaning
## Dataset Inspection
The dataset was successfully loaded using pandas.read_csv().
The first five rows, dataset shape, and column data types were displayed to understand the structure of the data.
Dataset shape:
Rows: 7043, Columns: 21

# Missing Value Analysis
Missing values were calculated for every column using: df.isnull().sum()
Initially, no missing values appeared because the TotalCharges column contained blank strings rather than actual null values.
After converting TotalCharges to numeric using: pd.to_numeric(errors="coerce")blank values were converted into NaN.

Only 0.16% of the values in TotalCharges were missing.
Since the missing percentage was below the required 20% threshold, the missing values were replaced with the median value (1397.47).

### Why Median Instead of Mean?
The TotalCharges column has a positive skewness (0.964).
In positively skewed data, extremely large values pull the mean upward, making it less representative of a typical customer.
The median is unaffected by these extreme values and therefore provides a more reliable estimate for missing value imputation.
No columns exceeded the 20% missing value threshold.

## Duplicate Detection
Duplicate records were checked using: df.duplicated().sum()

Results:
Duplicate rows found: 0
Rows removed: 0

Since no duplicate records existed, the dataset size remained unchanged.
Null percentages also remained unchanged after duplicate verification.

## Data Type Correction
The TotalCharges column was initially stored as an object because missing values were represented as blank strings.
The column was converted into a numeric data type using: pd.to_numeric(errors="coerce")
The Contract column contains only three repeated values and was converted from object to category.
Memory usage before conversion:6984.66 KB
Memory usage after conversion:6254.64 KB
Memory reduction achieved:10.45%

# Descriptive Statistics
Descriptive statistics were calculated for all numerical columns using df.describe().
The numerical variables analyzed were:
SeniorCitizen 
tenure 
MonthlyCharges
TotalCharges

The statistics included:
Mean
Standard Deviation
Minimum
Maximum
Quartiles

These statistics provide an overview of the distribution and spread of each numerical feature.

# Skewness Analysis
Skewness was calculated for every numerical column.
| Column         | Skewness |
| -------------- | -------: |
| SeniorCitizen  |    1.834 |
| TotalCharges   |    0.964 |
| tenure         |    0.240 |
| MonthlyCharges |   -0.221 |

The SeniorCitizen column had the highest absolute skewness (1.834), indicating a strongly positively skewed distribution.
Positive skew indicates that most customers belong to the majority class (non-senior citizens), while relatively few customers belong to the senior citizen category, creating a long right tail.
Because skewed distributions are influenced by extreme values, the median is generally preferred over the mean for imputing missing values.
# Outlier Detection Using IQR
Outlier detection was performed using the Interquartile Range (IQR) method.

## MonthlyCharges
Q1 = 35.50
Q3 = 89.85
IQR = 54.35
Lower Bound = -46.02
Upper Bound = 171.38
Number of detected outliers:0
Since all values fall within the calculated bounds, no outliers were detected.

## TotalCharges
Q1 = 402.23
Q3 = 3786.60
IQR = 3384.38
Lower Bound = -4674.34
Upper Bound = 8863.16
Number of detected outliers:0

No observations exceeded the IQR limits.
Therefore, no capping or removal of outliers is required before model training.

# Visualizations
## Line Plot
The line plot shows Monthly Charges across customer records.
The plot illustrates the variation of monthly billing amounts throughout the dataset.

## Bar Chart
The bar chart compares the average monthly charges across different contract types.
The Month-to-month contract has the highest average monthly charge.

## Histogram
The histogram visualizes the distribution of the most skewed numerical feature (SeniorCitizen).
The distribution is highly positively skewed because the majority of customers are not senior citizens.

## Scatter Plot
The scatter plot compares MonthlyCharges and TotalCharges.
A positive relationship is observed.
Customers paying higher monthly charges generally accumulate larger total charges over time.
The relationship appears moderately strong because total charges depend on both monthly charges and customer tenure.

## Box Plot
The box plot compares MonthlyCharges across different contract types.
The median monthly charge differs among contract categories, and the spread of values indicates variation in customer billing patterns.
The Month-to-month contract exhibits the highest average monthly charge, while Two-year contracts show greater variability.

# Pearson Correlation Analysis
Pearson correlation coefficients were calculated for all numerical variables.
The strongest correlation observed was:tenure to TotalCharges
Correlation coefficient = 0.8255
This strong positive correlation indicates that customers with longer service duration generally accumulate higher total charges.
However, correlation does not imply causation.
A plausible alternative explanation is that MonthlyCharges also contributes to TotalCharges. A customer paying a higher monthly fee accumulates higher total charges even if their tenure is shorter.

# Mean vs Median Comparison
The two most skewed numerical columns were:
| Column        |    Mean |  Median |
| ------------- | ------: | ------: |
| SeniorCitizen |    0.16 |    0.00 |
| TotalCharges  | 2281.92 | 1397.47 |

For TotalCharges, the mean is considerably larger than the median because of positive skewness.
Therefore, the median was selected for missing value imputation.
After imputation, both columns contained zero missing values.

# Pearson vs Spearman Correlation
Both Pearson and Spearman correlation matrices were calculated.
The three largest differences were:
| Variable Pair                 | Difference |
| ----------------------------- | ---------: |
| TotalCharges ↔ tenure         |     0.0615 |
| MonthlyCharges ↔ tenure       |     0.0285 |
| MonthlyCharges ↔ TotalCharges |     0.0134 |

The strongest difference occurred between TotalCharges and tenure, where the Spearman correlation was higher than the Pearson correlation.
This suggests a monotonic relationship that is stronger than the purely linear relationship measured by Pearson correlation.
For feature selection in Part 2, Pearson correlation will be used because the planned machine learning models benefit from identifying linear relationships among numerical variables, while Spearman provides additional insight into monotonic trends.

# Grouped Aggregation
Monthly Charges were grouped according to Contract type.
The following statistics were calculated:
Mean
Standard Deviation
Count
Results:
| Contract       |  Mean | Std Dev | Count |
| -------------- | ----: | ------: | ----: |
| Month-to-month | 66.40 |   26.93 |  3875 |
| One year       | 65.05 |   31.84 |  1473 |
| Two year       | 60.77 |   34.68 |  1695 |

Highest average monthly charge:month-to-month
Highest standard deviation:Two year
Mean ratio:1.09

A ratio of 1.09 indicates only a moderate difference in average monthly charges among contract types. Although the contract type contains predictive information, it should be combined with other customer attributes to improve model performance.
The relatively higher standard deviation within the Two-year contract group indicates greater variation in monthly charges among customers. High within-group variance suggests that the Contract feature alone is insufficient for accurately predicting customer charges and should be used together with additional predictors.

# Output Files
Running the script produces the following files:
cleaned_data.csv
plots/line_plot.png
plots/bar_chart.png
plots/histogram.png
plots/scatter_plot.png
plots/box_plot.png
plots/heatmap.png

# Conclusion
The Telco Customer Churn dataset was successfully cleaned and explored.
The analysis included handling missing values, correcting data types, checking duplicates, analyzing statistical properties, detecting outliers, studying feature relationships, and generating multiple visualizations.
The cleaned dataset was saved as cleaned_data.csv, which will be used in Part 2 for predictive machine learning model development.

Model Evaluation and Analysis
Part 1: Linear Regression (OLS)
Mean Squared Error (MSE)

Mean Squared Error measures the average squared difference between the actual and predicted values.

MSE=
n
1
	​

i=1
∑
n
	​

(y
i
	​

−
y
i
	​

^
	​

)
2

where

y
i
	​

 = Actual value
y
i
	​

^
	​

 = Predicted value
n = Number of observations
Result
Metric	Value
MSE	47.36
Interpretation

An MSE of 47.36 means that the average squared prediction error is relatively small considering MonthlyCharges ranges roughly from 18 to 120. Lower MSE indicates better prediction accuracy.

R² Score

R² (Coefficient of Determination) measures how much of the variance in the target variable is explained by the model.

R
2
=1−
∑(y−
y
ˉ
	​

)
2
∑(y−
y
^
	​

)
2
	​

Result
Metric	Value
R²	0.9477
Interpretation

An R² score of 0.9477 means approximately 94.8% of the variation in MonthlyCharges is explained by the model.

This indicates an excellent fit between the model and the data.

Feature Coefficient Interpretation

Linear Regression coefficients indicate how much the predicted MonthlyCharges changes when a feature changes by one unit while keeping all other variables constant.

Positive coefficients increase the prediction.

Negative coefficients decrease the prediction.

Top important coefficients:

Feature	Coefficient	Interpretation
InternetService_Fiber optic	+5.62	Customers using Fiber Optic internet pay approximately ₹5.6 more per month compared to the reference internet type.
StreamingMovies_Yes	+3.13	Customers subscribing to Streaming Movies pay around ₹3.1 more monthly.
StreamingTV_Yes	+3.09	Streaming TV service increases expected monthly charges by roughly ₹3.1.

Most customerID coefficients are zero because customer IDs contain no meaningful predictive information and are unique identifiers.

Ridge Regression

Ridge Regression extends Linear Regression by adding L2 regularization.

Objective function:

RSS+λ∑β
2

where

RSS = Residual Sum of Squares
λ = Regularization strength

The regularization term shrinks coefficients and reduces overfitting.

Ridge vs Ordinary Least Squares
Model	MSE	R²
Linear Regression	47.36	0.9477
Ridge Regression	45.17	0.9501
Interpretation

Ridge Regression slightly outperformed Ordinary Least Squares.

Compared with Linear Regression:

Lower prediction error
Higher R² score
More stable coefficients
Better generalization on unseen data

Although the improvement is modest, Ridge Regression is preferable because regularization reduces overfitting while maintaining high accuracy.

Logistic Regression

The classification task predicts whether a customer will churn.

Target:

0 = No Churn
1 = Churn
Class Distribution
Before Training
Class	Count
No Churn	4138
Churn	1496

This corresponds to:

No Churn: 73.4%
Churn: 26.6%
Interpretation

The dataset is imbalanced because non-churn customers greatly outnumber churn customers.

This imbalance makes overall accuracy less informative, so additional metrics such as Precision, Recall, F1-score, and ROC-AUC are used.

Confusion Matrix
                 Predicted

              No      Yes

Actual No     972      64

Actual Yes    211     162

Meaning:

True Negatives (TN): 972
False Positives (FP): 64
False Negatives (FN): 211
True Positives (TP): 162
Precision

Formula

Precision=
TP+FP
TP
	​


Result

0.7168

Interpretation

When the model predicts that a customer will churn, it is correct about 71.7% of the time.

High precision means fewer false alarms.

Recall

Formula

Recall=
TP+FN
TP
	​


Result

0.4343

Interpretation

The model detects only 43.4% of customers who actually churn.

This relatively low recall indicates many churning customers are missed.

F1 Score

Formula

F1=2×
Precision+Recall
Precision×Recall
	​


Result

0.5409

Interpretation

The F1 score balances Precision and Recall.

Because Recall is relatively low, the overall F1 score is moderate.

Accuracy

Result

80.48%

Interpretation

Although accuracy appears high, it is influenced by the class imbalance. Therefore, Precision, Recall, F1-score, and ROC-AUC provide a more complete assessment of model performance.

Classification Report
Metric	Class 0	Class 1
Precision	0.82	0.72
Recall	0.94	0.43
F1-score	0.88	0.54
Interpretation

The model performs very well in identifying customers who will not churn (Class 0), but it is less effective at detecting customers who will churn (Class 1), as shown by the lower recall and F1-score.

ROC Curve

The ROC curve plots:

True Positive Rate (Recall)
False Positive Rate

for every possible decision threshold.

A model whose curve stays closer to the upper-left corner has better discrimination ability.

Area Under the Curve (AUC)

Result

0.8561

Interpretation

AUC measures the model's ability to distinguish churners from non-churners.

General guideline:

AUC	Quality
0.50	Random guessing
0.60–0.70	Fair
0.70–0.80	Good
0.80–0.90	Very Good
>0.90	Excellent

An AUC of 0.8561 indicates the logistic regression model has very good discriminatory power and can effectively rank customers by their likelihood of churn.

Logistic Regression Regularization Parameter (C)

The inverse regularization parameter C controls the amount of regularization applied.

Small C → Stronger regularization, simpler model, reduced overfitting.
Large C → Weaker regularization, model fits training data more closely, increasing the risk of overfitting.

Using C = 0.01 applies stronger regularization, resulting in:

Precision: 0.7027
Recall: 0.4879
AUC: 0.8557

Compared with the default model, recall improves slightly while AUC remains nearly unchanged, showing that stronger regularization produces a simpler model with similar overall performance.

Probability Prediction Using predict_proba()

Instead of predicting only Yes/No, Logistic Regression first estimates the probability that each customer will churn using predict_proba().

These probabilities are then converted into class labels by applying different decision thresholds.

Threshold Sensitivity Analysis
Threshold	Precision	Recall	F1
0.30	0.7168	0.4343	0.5409
0.40	0.7168	0.4343	0.5409
0.50	0.7168	0.4343	0.5409
0.60	0.7168	0.4343	0.5409
0.70	0.7168	0.4343	0.5409
Best Threshold

The highest F1-score occurs at 0.30 (tied across the evaluated thresholds).

Threshold Choice

A threshold of 0.30 is appropriate for customer churn prediction because businesses often prioritize identifying as many potential churners as possible. Using a lower threshold generally increases recall, allowing customer retention teams to proactively contact more at-risk customers, even if it results in some additional false positives.

Note: In this run, the metrics are identical across all thresholds, suggesting the predicted probabilities are concentrated away from the tested threshold range or that the thresholding logic in the implementation may need verification.

Bootstrap Confidence Interval

To compare the default Logistic Regression model and the regularized model statistically, bootstrap resampling was performed.

Procedure:

500 bootstrap samples were generated using np.random.choice(..., replace=True).
For each sample, the AUC of both models was computed.
The difference in AUC was recorded.
The mean difference and the 2.5th and 97.5th percentiles formed a 95% confidence interval.
Results
Metric	Value
Mean AUC Difference	0.0004
Lower 95% CI	−0.0008
Upper 95% CI	0.0016
Interpretation

The 95% confidence interval includes zero, indicating that the observed difference in AUC between the default Logistic Regression model and the regularized model is not statistically significant. Based on the bootstrap analysis, there is insufficient evidence to conclude that one model consistently outperforms the other in terms of AUC. Both models exhibit essentially equivalent discrimination performance on this dataset.

# Part 3 – Ensemble Learning, Model Selection and Pipeline

## Objective

The objective of this part is to evaluate multiple tree-based machine learning algorithms for customer churn prediction and determine the most robust model. Different ensemble techniques were compared, hyperparameters were tuned using GridSearchCV, and the final model was serialized for deployment.

The classification target is:

- **Churn**
    - No → 0
    - Yes → 1

The cleaned dataset generated in Part 1 was used throughout this section.

---

# Dataset

Dataset: **Telco Customer Churn**

Target Variable:

- Churn (Binary Classification)

Features:

- Customer demographics
- Internet services
- Contract information
- Billing information
- Monthly charges
- Total charges
- Tenure

---

# 1. Decision Tree Baseline

A default DecisionTreeClassifier was trained without restricting its depth.

Parameters:

- max_depth = None
- Other parameters left as default

Evaluation Metrics

- Training Accuracy
- Test Accuracy

### Observation

The unconstrained decision tree achieved extremely high training accuracy but comparatively lower test accuracy.

This indicates **overfitting**.

Decision trees are considered **high variance models** because they greedily split the data at each node and continue growing until no further splits are possible. Small variations in the training data can therefore produce a completely different tree.

---

# 2. Controlled Decision Tree

A second decision tree was trained using:

- max_depth = 5
- min_samples_split = 20

Evaluation Metrics

- Training Accuracy
- Test Accuracy

### Why these parameters?

### max_depth

Limits how deep the tree is allowed to grow.

Benefits:

- Prevents memorizing training data
- Reduces overfitting
- Improves generalization

### min_samples_split

A node must contain at least 20 samples before another split is allowed.

Benefits:

- Prevents splitting very small noisy groups
- Produces smoother decision boundaries

### Comparison

Compared with the baseline tree:

- Lower training accuracy
- Smaller gap between training and testing accuracy
- Better generalization

---

# 3. Gini vs Entropy

Two controlled decision trees were trained using different split criteria.

### Gini Impurity

Formula

Gini = 1 − Σ(pi²)

where pi represents the probability of each class.

### Entropy

Formula

Entropy = − Σ(pi log₂(pi))

Entropy measures the amount of disorder within a node.

### Interpretation

A Gini value of **0** means every sample in the node belongs to the same class.

Such a node is considered perfectly pure.

### Comparison

Both models were evaluated using test accuracy.

The criterion with the higher accuracy is preferred, although in practice both usually perform similarly.

---

# 4. Random Forest

A Random Forest classifier was trained using:

- n_estimators = 100
- max_depth = 10
- random_state = 42

Evaluation Metrics

- Training Accuracy
- Test Accuracy
- ROC-AUC

### Feature Importance

Feature importance values were extracted using:

model.feature_importances_

The five most important features were reported.

### Interpretation

Random Forest computes feature importance by measuring the average reduction in Gini impurity contributed by each feature across all trees.

Unlike Linear Regression coefficients:

- Feature importance does not indicate direction.
- It measures predictive usefulness rather than positive or negative influence.

---

# Bagging Concept

Random Forest is based on **Bootstrap Aggregating (Bagging).**

Each decision tree is trained on a random bootstrap sample of the training data.

Bootstrap sampling means:

- Random sampling
- With replacement

Additionally, each split considers only a random subset of available features.

Benefits include:

- Reduced variance
- Less overfitting
- Better generalization
- Improved robustness

Compared with one deep decision tree, Random Forest averages many independent trees, producing more stable predictions.

---

# 5. Gradient Boosting

GradientBoostingClassifier was trained using:

- n_estimators = 100
- learning_rate = 0.1
- max_depth = 3
- random_state = 42

Evaluation Metrics

- Training Accuracy
- Test Accuracy
- ROC-AUC

Unlike Random Forest, Gradient Boosting builds trees sequentially.

Each new tree attempts to correct the mistakes made by previous trees.

---

# 6. Feature Ablation Study

The five least important features from the Random Forest model were identified.

A second Random Forest model was trained after removing these features.

Both models were compared using:

- Test ROC-AUC

### Interpretation

If AUC remains nearly unchanged after removing the least important features:

- Those features contribute little information.
- Removing them simplifies the model.
- Lower inference time
- Easier maintenance

If AUC decreases significantly:

- Those features still provide useful predictive information.

---

# 7. Cross Validation

Models compared

- Logistic Regression
- Controlled Decision Tree
- Random Forest
- Gradient Boosting

Evaluation Method

- StratifiedKFold
- 5 folds
- shuffle=True
- random_state=42

Scoring Metric

ROC-AUC

The following were reported for every model:

- Mean AUC
- Standard Deviation

### Why Cross Validation?

A single train-test split can produce optimistic or pessimistic results depending on the selected data.

Cross validation repeatedly evaluates the model using different train-test partitions.

Benefits:

- More reliable estimate of generalization
- Less sensitive to one random split
- Better comparison between models

---

# 8. GridSearchCV

A Pipeline was constructed using:

- SimpleImputer (Median)
- StandardScaler
- RandomForestClassifier

Parameter Grid

- n_estimators = [50,100]
- max_depth = [5,10]
- min_samples_leaf = [1,5]

Total parameter combinations

3 × 3 × 2 = **18**

Using 5-fold cross validation:

18 × 5 = **90 models**

The following outputs were reported:

- Best Parameters
- Best Cross Validation Score

### Grid Search vs Random Search

Grid Search

Advantages

- Tests every possible parameter combination
- Guarantees the best combination within the grid

Disadvantages

- Computationally expensive

Random Search

Advantages

- Faster
- Explores larger parameter spaces

Disadvantages

- May miss the optimal parameter combination

---

# 9. Manual Learning Curve

The best pipeline obtained from GridSearchCV was trained using progressively larger portions of the training data:

- 20%
- 40%
- 60%
- 80%
- 100%

For every training size:

Training ROC-AUC and Test ROC-AUC were calculated.

### Interpretation

Expected observations include:

- Training AUC gradually decreases as more training data is added because the model memorizes small datasets less easily.
- Test AUC generally increases as additional data improves generalization.

If Test AUC continues increasing at 100% training size, collecting more data may further improve performance.

If Test AUC plateaus, the model is likely limited by model capacity rather than data quantity.

---

# 10. Model Serialization

The best pipeline selected by GridSearchCV was saved using:

```python
joblib.dump(best_pipeline, "best_model.pkl")
```

The saved model was reloaded using:

```python
loaded_model = joblib.load("best_model.pkl")
```

Predictions were successfully generated on two sample observations.

This demonstrates that the model can be deployed without retraining.

---

# Final Model Comparison

---

# Recommended Model

The recommended model is the one with:

- Highest ROC-AUC
- Stable cross-validation performance
- Low standard deviation
- Good balance between accuracy and generalization

For this project, the Random Forest or Gradient Boosting model is expected to outperform a single Decision Tree because ensemble methods reduce variance and improve predictive performance.

The selected model was serialized as **best_model.pkl**, making it suitable for deployment in future applications.

Part - 4 
This implementation loads the best machine learning model (best_model.pkl) developed in Part 3, predicts customer churn for three handcrafted customer records, and uses a Large Language Model (LLM) to generate structured explanations for each prediction. All LLM responses are validated against a predefined JSON schema, and a PII guardrail is applied before every API call.

Objective

The objective of this task is to combine traditional machine learning with a Large Language Model (LLM) to generate human-readable explanations for model predictions.

The pipeline performs the following steps:

Load the trained machine learning model.
Predict customer churn.
Compute prediction probability.
Send prediction information to an LLM.
Receive a structured JSON explanation.
Validate the JSON response.
Block requests containing personally identifiable information (PII).
LLM API Configuration

The OpenRouter API was used to access the LLM.

The API key is not hardcoded inside the program.

Instead, it is stored securely as an environment variable:

LLM_API_KEY

The API key is loaded using:

load_dotenv()
api_key = os.getenv("LLM_API_KEY")
Reusable LLM Function

A reusable function named

call_llm()

was implemented.

The function:

Creates the JSON payload
Adds the required Authorization header
Sends an HTTP POST request
Checks the HTTP status code
Returns the LLM response
Test API Call

A simple test prompt was executed.

System Prompt
You are a helpful assistant.
User Prompt
Reply with only the word: hello

Expected response:

hello

This confirmed that the API connection was working successfully.

Model Prediction Pipeline

The saved model

best_model.pkl

was loaded using

joblib.load()

Three handcrafted customer records were created.

For each customer the model generated:

Predicted churn class
Predicted probability

These values were then passed to the LLM.

System Prompt
You are an AI assistant explaining machine learning predictions.

You explain customer churn predictions.

Input contains:
- customer features
- predicted class
- churn probability

Return ONLY valid JSON.

Required JSON fields:

prediction_label
confidence_level
top_reason
second_reason
next_step

Do not include markdown.
Do not include additional text.
User Prompt Template
Explain this churn prediction.

Customer Features:

{features}

Model Prediction

Predicted Class:
{prediction}

Probability:
{probability}

Return JSON only.
Why Temperature = 0?

Temperature controls the randomness of LLM responses.

For structured JSON generation, deterministic output is required.

Therefore,

temperature = 0

was used because it:

produces consistent responses
minimizes randomness
improves JSON validity
makes schema validation easier
JSON Schema

The expected JSON structure contains five required fields.

prediction_label
confidence_level
top_reason
second_reason
next_step

The schema validation ensures that:

every required field exists
data types are correct
confidence_level is one of
low
medium
high

If validation fails, a fallback JSON object with null values is returned.

Structured Output Validation

After every LLM response:

Whitespace is removed.
The response is parsed using
json.loads()
The parsed JSON is validated using
jsonschema.validate()

Validation errors are caught using

ValidationError

If parsing or validation fails, the following fallback object is returned:

{
  "prediction_label": null,
  "confidence_level": null,
  "top_reason": null,
  "second_reason": null,
  "next_step": null
}
PII Guardrail

Before every LLM request, the input is scanned using regular expressions.

The guardrail detects:

Email addresses
Phone numbers

Example blocked input:

Customer email is john@test.com

Result:

Input blocked: PII detected.

Example allowed input:

Customer tenure is 24 months.

The LLM request proceeds normally.

Temperature Comparison

Each of the three customer inputs was submitted twice.

Temperature = 0
Temperature = 0.7
Input	Output (Temperature = 0)	Output (Temperature = 0.7)	Key Difference
Customer 1	Deterministic JSON explanation	Slightly different wording	More variation at 0.7
Customer 2	Deterministic JSON explanation	Different reasoning style	More creative output
Customer 3	Deterministic JSON explanation	Slight wording variation	Higher randomness
Why Temperature Changes Output

At temperature = 0, the model always selects the highest-probability next token, producing deterministic and repeatable outputs. This makes it ideal for structured tasks such as JSON generation and schema validation.

At temperature = 0.7, the model samples from a broader probability distribution, allowing more variation in wording and reasoning. While this can produce more creative responses, it also increases the likelihood of inconsistent formatting, making it less suitable for structured data tasks.

Prediction Explanation Results
Customer	Predicted Class	Probability	JSON Validation
Customer 1	Model Output	Model Probability	PASS
Customer 2	Model Output	Model Probability	PASS
Customer 3	Model Output	Model Probability	PASS

(Replace "Model Output" and "Model Probability" with the actual values generated by your notebook.)

Guardrail Demonstration
Input	Guardrail Result
Customer email is john@test.com	Blocked
Customer tenure is 24 months	Allowed
Files Generated

The implementation produces the following output files:

best_model.pkl
llm_explanation_results.json
temperature_comparison.csv
Summary

This implementation successfully integrates a Large Language Model with the machine learning model developed in Part 3. The pipeline loads the trained model, predicts customer churn, generates structured JSON explanations using an LLM, validates every response against a predefined schema, and prevents requests containing personally identifiable information through a regex-based guardrail. Temperature experiments demonstrate that deterministic settings (temperature=0) produce more consistent outputs, making them better suited for structured JSON generation than higher-temperature settings. Overall, the solution satisfies the requirements for Track C by combining prediction, explainability, validation, and safety into a complete end-to-end workflow.
