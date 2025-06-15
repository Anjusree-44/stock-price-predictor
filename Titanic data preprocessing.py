# Import necessary libraries
import pandas as pd
print("Pandas version:", pd.__version__)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
import os
print("Current working directory:", os.getcwd())
# Load the dataset
df = pd.read_csv("C:/Users/katta/OneDrive/Python-ai/Titanic.csv")  # Ensure Titanic.csv is in your working directory
# 1. Explore the data
print("First 5 rows of the dataset:")
print(df.head())

print("\nData Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# 2. Handle missing values
# Fill Age with median, Cabin with 'Unknown', Embarked with mode
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Cabin'].fillna('Unknown', inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

print("\nMissing Values After Imputation:")
print(df.isnull().sum())

# 3. Convert categorical variables to numeric
# Drop Ticket and Name for simplicity
df.drop(['Ticket', 'Name'], axis=1, inplace=True)

# Label encode 'Sex'
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])  # male: 1, female: 0

# One-hot encode 'Embarked'
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Optional: Simplify Cabin (use just first letter)
df['Cabin'] = df['Cabin'].apply(lambda x: x[0] if x != 'Unknown' else 'U')
df = pd.get_dummies(df, columns=['Cabin'], drop_first=True)

# 4. Normalize/Standardize numerical features
# MinMax Scaling Age and Fare
scaler = MinMaxScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# 5. Visualize outliers using boxplots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(data=df, y='Age')
plt.title("Boxplot of Age")

plt.subplot(1, 2, 2)
sns.boxplot(data=df, y='Fare')
plt.title("Boxplot of Fare")

plt.tight_layout()
plt.show()

# Remove outliers using IQR method (optional but recommended)
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

df = remove_outliers_iqr(df, 'Fare')
df = remove_outliers_iqr(df, 'Age')

print("\nCleaned Data Shape:", df.shape)
print("\nFinal Preview of Cleaned Data:")
print(df.head())

# Save cleaned data
df.to_csv("cleaned_titanic-dataset.csv", index=False)
