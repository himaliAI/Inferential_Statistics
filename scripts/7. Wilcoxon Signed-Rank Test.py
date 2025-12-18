import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

df = pd.read_csv("D:/IDA/data/adult.csv")

num_cols = df.select_dtypes(include='number').columns.tolist()
num_cols.remove('education.num')
cat_cols = df.select_dtypes(include='object').columns.tolist()

# shuffle dataframe randomly, rest index, drop old index
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# split into two halves
half = len(shuffled_df) // 2 # floor division
group1 = shuffled_df.iloc[:half]['hours.per.week'].values
group2 = shuffled_df.iloc[half:half*2]['hours.per.week']

# manipulate group 2 to add after effect
group2 += 1

stat, p_value = wilcoxon(group1, group2, alternative='two-sided')

print(f"\nWilcoxon statistics: {stat:.4f}")
print(f"P-value: {p_value:.4f}")
