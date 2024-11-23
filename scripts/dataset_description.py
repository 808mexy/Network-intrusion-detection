import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('C:/Users/19562/Downloads/Thursday-WorkingHours-Morning-WebAttacks.csv')

# Print descriptive statistics
print(df.describe())

# Plot class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x=' Label', data=df)
plt.title("Class Distribution in the Dataset")
plt.xlabel("Class Labels")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Drop non-numeric columns before creating the correlation matrix
non_numeric_columns = ['Flow ID', ' Source IP', ' Destination IP', ' Timestamp', ' Label']
df_numeric = df.drop(columns=non_numeric_columns, errors='ignore')

# Compute and plot the correlation matrix for numeric data only
plt.figure(figsize=(12, 10))
correlation_matrix = df_numeric.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()
