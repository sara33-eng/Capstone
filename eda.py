"""
Capstone Project - Part 1
Exploratory Data Analysis (EDA)

"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
# Create output folders
os.makedirs("plots", exist_ok=True)
# Load Dataset
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
print("=" * 80)
print("Loading Dataset...")
print("=" * 80)
df = pd.read_csv(DATA_PATH)

# Basic Dataset Information
print("\nFirst Five Rows")
print(df.head())
print("\nDataset Shape")
print(df.shape)
print("\nData Types")
print(df.dtypes)

# Memory Usage Before Conversion
memory_before = df.memory_usage(deep=True).sum()
print("\nMemory Usage Before Type Conversion")
print(f"{memory_before/1024:.2f} KB")

# NULL VALUE ANALYSIS
print("\n" + "=" * 80)
print("NULL VALUE ANALYSIS")
print("=" * 80)
null_count = df.isnull().sum()
null_percent = (null_count / len(df)) * 100
null_table = pd.DataFrame({ "Null Count": null_count, "Null Percentage": null_percent})
print(null_table.sort_values(by="Null Percentage",ascending=False))
print("\nColumns having more than 20% missing values:")
high_null_columns = null_table[null_table["Null Percentage"] > 20]
if len(high_null_columns) == 0:
    print("None")
else:
    print(high_null_columns)

# DATA TYPE CORRECTION
print("\n" + "=" * 80)
print("DATA TYPE CORRECTION")
print("=" * 80)
print("\nDatatype before conversion")
print(df["TotalCharges"].dtype)

# TotalCharges contains blank spaces, convert blanks to NaN and then numeric.
df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
df["TotalCharges"] = pd.to_numeric(    df["TotalCharges"],errors="coerce")
print("Datatype after conversion")
print(df["TotalCharges"].dtype)

# Convert repetitive string column to category
print("\nContract datatype before conversion")
print(df["Contract"].dtype)
df["Contract"] = df["Contract"].astype("category")
print("Contract datatype after conversion")
print(df["Contract"].dtype)

# FILL NUMERIC COLUMNS (<20% NULLS)
print("\n" + "=" * 80)
print("MEDIAN IMPUTATION")
print("=" * 80)
numeric_columns = df.select_dtypes(include=np.number).columns
for column in numeric_columns:
    missing_percent = df[column].isnull().mean() * 100
    if missing_percent < 20:
        if df[column].isnull().sum() > 0:
            median_value = df[column].median()
            print(f"Filling '{column}' ({missing_percent:.2f}% missing) using median = {median_value:.2f}")
            df[column] = df[column].fillna(median_value)
print("\nRemaining Missing Values")
print(df[numeric_columns].isnull().sum())

# DUPLICATE ANALYSIS
print("\n" + "=" * 80)
print("DUPLICATE ANALYSIS")
print("=" * 80)
duplicate_rows = df.duplicated().sum()
print(f"Duplicate Rows Found : {duplicate_rows}")
rows_before = len(df)
df = df.drop_duplicates()
rows_after = len(df)
rows_removed = rows_before - rows_after
print(f"Rows Before : {rows_before}")
print(f"Rows After  : {rows_after}")
print(f"Rows Removed: {rows_removed}")

# NULL PERCENTAGE AFTER DUPLICATE REMOVAL
print("\nNull Percentage After Removing Duplicates")
new_null = (df.isnull().sum() / len(df)) * 100
null_after = pd.DataFrame({"Null Percentage": new_null})
print(null_after.sort_values( by="Null Percentage",ascending=False))

# MEMORY AFTER TYPE CONVERSION
memory_after = df.memory_usage(deep=True).sum()
print("\nMemory Usage After Type Conversion")
print(f"{memory_after/1024:.2f} KB")
reduction = ((memory_before - memory_after)/ memory_before) * 100
print(f"Memory Reduction : {reduction:.2f}%")
print("\nData Cleaning Completed Successfully.")

# DESCRIPTIVE STATISTICS
print("\n" + "=" * 80)
print("DESCRIPTIVE STATISTICS")
print("=" * 80)
numeric_df = df.select_dtypes(include=np.number)
print("\nSummary Statistics")
print(numeric_df.describe())

# SKEWNESS ANALYSIS
print("\n" + "=" * 80)
print("SKEWNESS ANALYSIS")
print("=" * 80)
skewness = numeric_df.skew().sort_values(key=lambda x: x.abs(),ascending=False)
print(skewness)
most_skewed_column = skewness.index[0]
print(f"\nMost Skewed Column : {most_skewed_column}")
print(f"Skewness Value     : {skewness.iloc[0]:.3f}")

# IQR OUTLIER DETECTION
print("\n" + "=" * 80)
print("IQR OUTLIER DETECTION")
print("=" * 80)

# Two numeric columns for analysis
iqr_columns = ["MonthlyCharges","TotalCharges"]
for column in iqr_columns:
    print(f"\nAnalyzing Column : {column}")
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    outliers = df[(df[column] < lower_bound) |(df[column] > upper_bound)]
    print(f"Q1: {Q1:.2f}")
    print(f"Q3: {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"Lower Bound: {lower_bound:.2f}")
    print(f"Upper Bound: {upper_bound:.2f}")
    print(f"Outlier Count: {len(outliers)}")

# VISUALIZATION SETTINGS
plt.style.use("ggplot")

# LINE PLOT
print("\nGenerating Line Plot...")
plt.figure(figsize=(12, 6))
plt.plot(df.index,df["MonthlyCharges"],linewidth=1)
plt.title("Monthly Charges Across Dataset")
plt.xlabel("Customer Index")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig("plots/line_plot.png",dpi=300)
plt.close()
print("Saved -> plots/line_plot.png")

# BAR CHART
print("\nGenerating Bar Chart...")
avg_monthly = (df.groupby("Contract")["MonthlyCharges"].mean().sort_values())
plt.figure(figsize=(8, 5))
plt.bar(avg_monthly.index.astype(str),avg_monthly.values)
plt.title("Average Monthly Charges by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Average Monthly Charges")
plt.tight_layout()
plt.savefig("plots/bar_chart.png",dpi=300)
plt.close()
print("Saved -> plots/bar_chart.png")

# HISTOGRAM
print("\nGenerating Histogram...")
plt.figure(figsize=(10, 6))
sns.histplot(df[most_skewed_column],bins=20,kde=True)
plt.title(f"Distribution of {most_skewed_column}")
plt.xlabel(most_skewed_column)
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("plots/histogram.png",dpi=300)
plt.close()
print("Saved -> plots/histogram.png")

# SCATTER PLOT
print("\n" + "=" * 80)
print("SCATTER PLOT")
print("=" * 80)
plt.figure(figsize=(8,6))
sns.scatterplot(data=df,x="MonthlyCharges",y="TotalCharges",alpha=0.6)
plt.title("Monthly Charges vs Total Charges")
plt.xlabel("Monthly Charges")
plt.ylabel("Total Charges")
plt.tight_layout()
plt.savefig("plots/scatter_plot.png",dpi=300)
plt.close()
print("Scatter plot saved.")

# BOX PLOT
print("\n" + "=" * 80)
print("BOX PLOT")
print("=" * 80)
plt.figure(figsize=(8,6))
sns.boxplot(data=df,x="Contract",y="MonthlyCharges")
plt.title("Monthly Charges by Contract Type")
plt.xlabel("Contract")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig("plots/box_plot.png",dpi=300)
plt.close()
print("Box plot saved.")

# PEARSON CORRELATION MATRIX
print("\n" + "=" * 80)
print("PEARSON CORRELATION MATRIX")
print("=" * 80)
numeric_df = df.select_dtypes(include=np.number)
pearson_corr = numeric_df.corr()
print(pearson_corr)

# CORRELATION HEATMAP
plt.figure(figsize=(10,8))
sns.heatmap(pearson_corr,annot=True,cmap="coolwarm",fmt=".2f")
plt.title("Pearson Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/heatmap.png",dpi=300)
plt.close()
print("Heatmap saved.")

# HIGHEST ABSOLUTE CORRELATION PAIR
print("\n" + "=" * 80)
print("HIGHEST CORRELATED VARIABLE PAIR")
print("=" * 80)
corr_abs = pearson_corr.abs().copy()
np.fill_diagonal(corr_abs.values, 0)
stacked = (corr_abs.stack().reset_index())
stacked.columns = ["Variable 1","Variable 2","Correlation"]

# Remove duplicate pairs (A,B) and (B,A)
stacked["Pair"] = stacked.apply(lambda row: tuple(sorted([row["Variable 1"], row["Variable 2"]])),axis=1)
stacked = stacked.drop_duplicates(subset="Pair")
highest_pair = stacked.sort_values(by="Correlation",ascending=False).iloc[0]
print("\nHighest Absolute Correlation")
print(f"Variable 1 : {highest_pair['Variable 1']}")
print(f"Variable 2 : {highest_pair['Variable 2']}")
print(f"Correlation: {highest_pair['Correlation']:.4f}")
print("\nVisualization section completed successfully.")

# Mean vs Median
print("\n" + "=" * 80)
print("MEAN VS MEDIAN COMPARISON")
print("=" * 80)
top2 = skewness.abs().sort_values(ascending=False).head(2).index
for col in top2:
    print(f"\nColumn : {col}")
    print(f"Mean   : {df[col].mean():.2f}")
    print(f"Median : {df[col].median():.2f}")
    df[col] = df[col].fillna(df[col].median())
print("\nNull Values After Imputation")
print(df[top2].isnull().sum())

#Spearman Correlation
print("\n" + "=" * 80)
print("PEARSON VS SPEARMAN CORRELATION")
print("=" * 80)
pearson = numeric_df.corr()
spearman = numeric_df.corr(method="spearman")
print("\nPearson Correlation Matrix")
print(pearson)
print("\nSpearman Correlation Matrix")
print(spearman)
diff = (spearman - pearson).abs()
pairs = []

for i in diff.columns:
    for j in diff.columns:
        if i < j:
            pairs.append([i, j, diff.loc[i, j]])
difference = pd.DataFrame(pairs,columns=["Column1","Column2","Difference"])
difference = difference.sort_values(by="Difference",ascending=False)
print("\nTop 3 Correlation Differences")
print(difference.head(3))

#GROUPED AGGREGATION
print("\n" + "=" * 80)
print("GROUPED AGGREGATION")
print("=" * 80)
group = df.groupby("Contract")["MonthlyCharges"].agg(["mean","std","count"])
print(group)
highest_mean = group["mean"].idxmax()
highest_std = group["std"].idxmax()
ratio = (group["mean"].max()/group["mean"].min())
print("\nHighest Mean Group")
print(highest_mean)
print("\nHighest Standard Deviation Group")
print(highest_std)
print(f"\nMean Ratio : {ratio:.2f}")

print("\nSaving cleaned dataset...")

df.to_csv("cleaned_data.csv",index=False)
print("cleaned_data.csv saved successfully.")
print("\nEDA Completed Successfully.")
