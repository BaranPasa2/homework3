import pandas as pd
import numpy as np
from causalinference import CausalModel
import statsmodels.api as sm

# Load and preprocess data
df = pd.read_pickle('submission1/data/output/TaxBurden_Data.pkl')
df['log_sales'] = np.log(df['sales_per_capita'])
df['log_price'] = np.log(df['cost_per_pack'])
df['log_tax'] = np.log(df['tax_dollar'])

# Function to run OLS and IV using causalinference
def run_ols_iv(start_year, end_year):
    sub_df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)].dropna()

    # Prepare arrays
    Y = sub_df['log_sales'].values
    D = sub_df['log_price'].values.reshape(-1, 1)
    Z = sub_df['log_tax'].values.reshape(-1, 1)

    # Center the variables
    Y_c = Y - Y.mean()
    D_c = D - D.mean()
    Z_c = Z - Z.mean()

    # CausalModel: IV estimation (tax → price → sales)
    causal = CausalModel(Y=Y_c, D=D_c, Z=Z_c)
    causal.est_via_2sls()

    # OLS Estimation (log_sales ~ log_price)
    D_with_const = sm.add_constant(D)
    ols_model = sm.OLS(Y, D_with_const).fit()

    # First-stage: log_price ~ log_tax
    Z_with_const = sm.add_constant(Z)
    first_stage = sm.OLS(D, Z_with_const).fit()

    # Reduced form: log_sales ~ log_tax
    reduced_form = sm.OLS(Y, Z_with_const).fit()

    # Outputs
    print(f"--- Period: {start_year}-{end_year} ---")
    print("OLS Estimate (Elasticity):", round(ols_model.params[1], 4))
    print("IV Estimate (Elasticity via tax instrument):", round(causal.est['ate'], 4))
    print("\nFirst Stage (log_price ~ log_tax):")
    print(first_stage.summary())
    print("\nReduced Form (log_sales ~ log_tax):")
    print(reduced_form.summary())
    print("----------------------------------\n")
    return ols_model.params[1], causal.est['ate']

# Prompt 6–8: Period 1970–1990
ols_early, iv_early = run_ols_iv(1970, 1990)

# Prompt 9: Period 1991–2015
ols_late, iv_late = run_ols_iv(1991, 2015)

# Prompt 10: Comparison
print("Elasticity Comparison:")
print(f"1970-1990 OLS Elasticity: {round(ols_early, 4)}, IV Elasticity: {round(iv_early, 4)}")
print(f"1991-2015 OLS Elasticity: {round(ols_late, 4)}, IV Elasticity: {round(iv_late, 4)}")

# Interpretation
if abs(iv_early - iv_late) > 0.1:
    print("\nElasticities differ significantly between periods.")
    print("Potential reasons: shifts in consumer behavior, increased awareness of health risks, or different effectiveness of taxes.")
else:
    print("\nElasticities are similar across periods.")
