import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

df = pd.read_csv("D:/IDA/data/adult.csv")
# challange with this dataset
    # It does not naturally contain before/after measures
    # So, we need to simulate it
'''
num_cols = df.select_dtypes(include='number').columns.tolist()
num_cols.remove('education.num')
cat_cols = df.select_dtypes(include='object').columns.tolist()
actual_hours = df['hours.per.week']
adjusted_hours = actual_hours - 2 # pretend after intervention

# paired t-test
t_stat, p_value = ttest_rel(actual_hours, adjusted_hours)
dof = len(actual_hours) - 1

print(f"\nMean before: {actual_hours.mean():.4f}")
print(f"Mean after: {adjusted_hours.mean():.4f}")
print(f"T-statistics: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
'''

# Paired t-test by splitting the df into matched subgroups
    # Randomly split dataset into two halves and compare their mean
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # .sample is pandas method to randomly select rows
    # frac=1 -> sample all rows randomly
    # random_state for same results each time
    # reset_index -> index is preserved but shuffled. So reset from 0, 1, 2, ....
    # drop=True -> prevent old index from being added as a new column
half = len(shuffled_df) // 2 # // -> floor division and returns integer
group_1 = shuffled_df['hours.per.week'][: half]
group_2 = shuffled_df['hours.per.week'][half:half*2 ]
print(f"\nFirst half: {len(group_1)}")
print(f"Second half: {len(group_2)}")
print(f"Total: {len(df)}")

# paired t-test
t_stats, p_vlaue = ttest_rel(group_1, group_2)
#dof = len(group_1) - 1

print(f"\nMean of Group 1: {group_1.mean():.4f}")
print(f"Mean of Group2: {group_2.mean():.4f}")
print(f"T-statistics: {t_stats:.4f}")
#print(f"DoF: {dof}")
print(f"P-value: {p_vlaue:.4f}")
