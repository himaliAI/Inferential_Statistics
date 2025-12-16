import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import norm

df = pd.read_csv("D:/IDA/data/adult.csv")

num_cols = df.select_dtypes(include='number').columns.tolist()
num_cols.remove('education.num')
cat_cols = df.select_dtypes(include='object').columns.tolist()

# Chi-square test between sex and income (<50k, >=50K)
# Create contingency table 
contingency = pd.crosstab(df['sex'], df['income'])
row_percentages = contingency.div(contingency.sum(axis=1), axis=0) * 100
contingency_with_pct = contingency.astype(str) + " (" + row_percentages.round(2).astype(str) + "%)"

chi2, p, dof, expected = chi2_contingency(contingency)
print(f"\nContingency table:\n{contingency_with_pct}")
print(f"\nTable of Expected Values:\n{np.round(expected, 2)}")
print(f"\nChi2: {chi2}\nDoF: {dof}\np-value: {p:.4f}")

'''# Chi-square test between education Vs income (7x2 table)
    # Then conduct post-hoc analysis by computing standardized residuals
    # followed by Bonferroni correction
contingency = pd.crosstab(df['education'], df['income'])
print(f"\nContingency Table:\n{contingency}")

chi2, p, dof, expected = chi2_contingency(contingency)
#print(f"\nChi-square:{chi2}\np-value: {p:.4f}")
# Chi2=4429, p<0.001

# compute standardized residuals (z-scores)
residuals = (contingency - expected) / np.sqrt(expected)
print(f"\nResiduals:\n{residuals}")

# convert residuals to p value
p_values = 2 * (1 - norm.cdf(np.abs(residuals)))

# Bonferroni correction
from statsmodels.stats.multitest import multipletests
# flatten p-values
p_values_flat = p_values.flatten()

reject, p_corrected, _, _ = multipletests(p_values_flat, alpha=0.05, method='bonferroni')

# reshape p back to table form
p_corrected_table = pd.DataFrame(p_corrected.reshape(contingency.shape), index=contingency.index, columns=contingency.columns)
reject_table = pd.DataFrame(reject.reshape(contingency.shape), index=contingency.index, columns=contingency.columns)

print(f"Corrected p-values (Bonferroni):\n{p_corrected_table}")
print(f"Significant cells (True):\n{reject_table}")
'''

'''# Fisher's exact tests for 2x2 table
from scipy.stats import fisher_exact # supports only 2x2 table
contingency = pd.crosstab(df['sex'], df['income'])
print(f"Contingency Table:\n{contingency}")

oddsratio, p_value = fisher_exact(contingency)
print(f"\nOdds ratio: {oddsratio:.4f}")
print(f"P-value: {p_value:.4f}")
'''
