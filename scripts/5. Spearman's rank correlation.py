import pandas as pd
import numpy as np
from scipy.stats import spearmanr

df = pd.read_csv("D:/IDA/data/adult.csv")

num_cols = df.select_dtypes(include='number').columns.tolist()
num_cols.remove('education.num')
cat_cols = df.select_dtypes(include='object').columns.tolist()

# relationship between age and hours.per.week assuming they are ordinal or severely skewed data
rho, p_value = spearmanr(df['age'], df['hours.per.week'])

print(f"\nSpearman correlation cofficient: {rho:.4f}")
print(f"P-vlaue: {p_value}")
