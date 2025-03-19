import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

# Load data
df = pd.read_pickle('submission2/data/output/TaxBurden_Data.pkl')
df['log_sales'] = np.log(df['sales_per_capita'])
df['log_price'] = np.log(df['cost_per_pack'])
df['log_tax'] = np.log(df['tax_dollar'])

def run_ols_iv(start_year, end_year):
    sub_df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)].dropna()

    # OLS Regression
    ols_model = sm.OLS(sub_df['log_sales'], sm.add_constant(sub_df['log_price'])).fit()

    # IV Regression: log_sales ~ log_price (instrumented by log_tax)
    iv_model = IV2SLS.from_formula(
        'log_sales ~ 1 + [log_price ~ log_tax]',
        data=sub_df
    ).fit()

    # First Stage: log_price ~ log_tax
    first_stage = sm.OLS(sub_df['log_price'], sm.add_constant(sub_df['log_tax'])).fit()

    # Reduced Form: log_sales ~ log_tax
    reduced_form = sm.OLS(sub_df['log_sales'], sm.add_constant(sub_df['log_tax'])).fit()

    # Output
    print(f"--- Period: {start_year}-{end_year} ---")
    print(f"OLS Elasticity Estimate: {round(ols_model.params['log_price'], 4)}")
    print(f"IV Elasticity Estimate: {round(iv_model.params['log_price'], 4)}")
    print("\nFirst Stage (log_price ~ log_tax):")
    print(first_stage.summary())
    print("\nReduced Form (log_sales ~ log_tax):")
    print(reduced_form.summary())
    print("----------------------------------\n")
    
    return ols_model.params['log_price'], iv_model.params['log_price']

# 1970–1990
ols_early, iv_early = run_ols_iv(1970, 1990)

# 1991–2015
ols_late, iv_late = run_ols_iv(1991, 2015)

# Compare Elasticities
print("Elasticity Comparison:")
print(f"1970-1990 OLS: {round(ols_early,4)}, IV: {round(iv_early,4)}")
print(f"1991-2015 OLS: {round(ols_late,4)}, IV: {round(iv_late,4)}")

if abs(iv_early - iv_late) > 0.1:
    print("\nElasticities differ significantly. This could be due to changing consumer responsiveness or policy impacts.")
else:
    print("\nElasticities are similar across periods.")
