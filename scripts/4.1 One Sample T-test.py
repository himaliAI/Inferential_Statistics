import pandas as pd
import numpy as np

df = pd.read_csv("D:/IDA/data/adult.csv")

num_cols = df.select_dtypes(include='number').columns.tolist()
num_cols.remove('education.num')
cat_cols = df.select_dtypes(include='object').columns.tolist()

# one sample t-test
# we are testing averge age of adults 
    # (mean = 38.58) is signifcantly different than 40?
from scipy.stats import ttest_1samp
t_stat, p_value = ttest_1samp(df['age'], 40)
dof = len(df) - 1 # number of sample minus one

print(f"T-statistics: {t_stat:.4f}")
print(f"DoF: {dof}")
print(f"P-value: {p_value:.4f}")
print(f"Mean age: {df['age'].mean():.2f}")
