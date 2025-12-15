import pandas as pd

df = pd.read_csv("D:/IDA/data/adult.csv")

num_cols = df.select_dtypes(include='number').columns.tolist()
num_cols.remove('education.num')
cat_cols = df.select_dtypes(include='object').columns.tolist()
print(f"\nNumerical Columns:\n{num_cols}")
print(f"\nCategorical Columns:\n{cat_cols}")