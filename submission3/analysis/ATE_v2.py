import pandas as pd
import numpy as np
import pyfixest as pf
# Load data
df = pd.read_pickle('submission2/data/output/TaxBurden_Data.pkl')

# Create log variables
df['log_sales'] = np.log(df['sales_per_capita'])
df['log_price'] = np.log(df['price_cpi'])
df['log_total_tax'] = np.log(df['tax_dollar']*(218/(df.index + 1)))
df['ln_price_cpi'] = np.log(df['price_cpi'])
#df['ln_state_tax'] = np.log(df['state_tax'])


#print(df['log_total_tax'])

# Filter datasets by year
df_70_90 = df[(df['Year'] >= 1970) & (df['Year'] <= 1990)].copy()
df_91_15 = df[(df['Year'] >= 1991) & (df['Year'] <= 2015)].copy()
#print(df_70_90.columns)
#print(df_70_90.index)

# -- Q 6 --
ols_70_90 = pf.feols('log_sales ~ log_price', data=df_70_90)
# Extract the coefficient for  and print it
print(f"{ols_70_90.coef()['log_price']:.4f}")
# Add new row with value for 'Coefficient', and 'Variable' as NaN or placeholder
#new_row = pd.DataFrame([{'Variable': 'Log Price', 'Coefficient': ols_70_90.coef()['log_price']}])
#coef_table = pd.concat([coef_table, new_row], ignore_index=True)

# -- Q 7 --
iv_70_90 = pf.feols('log_sales ~ 1 | log_price ~ log_total_tax', data=df_70_90)
print(iv_70_90.summary())
#print(f"{iv_70_90.coef()['log_price']:.4f}")

# -- Q 8 --
stage1 = pf.feols('log_price ~ tax_dollar', data=df_70_90)
price_hat = stage1.predict()
df_70_90['price_hat'] = price_hat
stage2 = pf.feols('log_sales ~ price_hat', data=df_70_90)
print(f"First Stage: {stage1.summary()}")
print(f"Second Stage: {stage2.summary()}")

# -- Q 9 --
ols_91_15 = pf.feols('log_sales ~ log_price', data=df_91_15)
iv_91_15 = pf.feols('log_sales ~ 1 | log_price ~ log_total_tax', data=df_91_15)
print(f"{ols_91_15.coef()['log_price']:.4f}")
print(f"{iv_91_15.coef()['log_price']:.4f}")


stage1 = pf.feols('log_price ~ log_total_tax', data=df_91_15)
price_hat = stage1.predict()
df_91_15['price_hat'] = price_hat
stage2 = pf.feols('log_sales ~ price_hat', data=df_91_15)
