import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from scipy.stats import levene

df = pd.read_csv("D:/IDA/data/adult.csv")

num_cols = df.select_dtypes(include='number').columns.tolist()
num_cols.remove('education.num')
cat_cols = df.select_dtypes(include='object').columns.tolist()

# do average hours.per.week differ between gender?
male_hours = df[df['sex'] == 'Male']['hours.per.week']
female_hours = df[df['sex'] == 'Female']['hours.per.week']
print(male_hours.head())

t_test, p_value = ttest_ind(male_hours, female_hours, equal_var=True)
dof = len(male_hours) + len(female_hours) - 2

print(f"\nMean hours (Male): {male_hours.mean():.2f}")
print(f"Mean hours (Female): {female_hours.mean():.2f}")
print(f"T-statistics: {t_test:.2f}")
print(f"DoF: {dof}")
print(f"P-value: {p_value:.4f}")

# Note: This is a classic student-t test where we assume equal variances between the groups
# Actually, we need to test for equality of variances
    # with Levene's test

# Test of equality of variance with levene's test
levene, p_levene = levene(male_hours, female_hours)
print(f"\nLevene's test: {levene:.4f}")
print(f"P-value: {p_levene:.4f}")
# if equal variance -> equal_var = True; else equal_var=False


