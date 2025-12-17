import pandas as pd
import numpy as np
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

df = pd.read_csv("D:/IDA/data/adult.csv")

num_cols = df.select_dtypes(include='number').columns.tolist()
num_cols.remove('education.num')
cat_cols = df.select_dtypes(include='object').columns.tolist()

# ANOVA for average hours worked per week across different education level
# group by education
groups = [group['hours.per.week'].values for name, group in df.groupby('education')]
    # df.groupby('education') -> split df to sub-dataframe according to values of 'education' column
    # Result has names (values of 'education') and groups (sub-dataframe with rows of only that group)
    # [df['hours.per.week'].values -> select 'hours.per.week' column of subgroup (group) and convert to np array (.values)
    # outer [] -> collect all those arrays into a single python list
    # Finally: groups becomes a list of arrays (one array per education category) 

# One-way ANOVA
f_stat, p_value = f_oneway(*groups)
    # * inside a function call unpacks a list or tuple into seperate arguments
        # equivalent to f_oneway(group1, group2, group3, ...., groupn) where each group is an array of a education value

print(f"\nF-statistics: {f_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Tukey's HSD (honest significant differences) test
    # for post-hoc analysis if p is significant
tukey = pairwise_tukeyhsd(endog=df['hours.per.week'], groups=df['education'], alpha=0.05)
    # endog -> endogenous (outcome/dependent) variable 
print(tukey)
