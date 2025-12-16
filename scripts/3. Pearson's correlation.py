import pandas as pd
import numpy as np
from scipy.stats import pearsonr

df = pd.read_csv("D:/IDA/data/adult.csv")

num_cols = df.select_dtypes(include='number').columns.tolist()
num_cols.remove('education.num')
cat_cols = df.select_dtypes(include='object').columns.tolist()

# Pearson's correlation between age and hours.per.week
r, p_value = pearsonr(df['age'], df['hours.per.week'])
dof = len(df['age']) - 2 # sample size - 2
print(f"Pearson R: {r:.4f}")
print(f"DoF: {dof}")
print(f"P-value: {p_value:.4f}")

# visualize correlations between numerical columns by sns.pairplot
import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(df[num_cols], diag_kind='kde', kind='reg') # kind='reg' adds regression line
plt.suptitle("Pairplots of numerical variables", y=1.01) 
plt.show()

# Correlation maxtrix between numerical columns
corr_matrix = df[num_cols].corr(method='pearson')
print(corr_matrix.round(3))

# Heatmap of correlation matrix
plt.figure(figsize=(10, 8))
plt.title("Pearson's correlation Heatmap")
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.show()