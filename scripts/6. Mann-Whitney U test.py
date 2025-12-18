import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

df = pd.read_csv("D:/IDA/data/adult.csv")

num_cols = df.select_dtypes(include='number').columns.tolist()
num_cols.remove('education.num')
cat_cols = df.select_dtypes(include='object').columns.tolist()

# compare hours.per.week between male and females
male_hours = df[df['sex'] == 'Male']['hours.per.week']
female_hours = df[df['sex'] == 'Female']['hours.per.week']

u_stat, p_value = mannwhitneyu(male_hours, female_hours, alternative='two-sided')

print(f"\nU-statistics: {u_stat}")
print(f"P-value: {p_value:.4f}")
