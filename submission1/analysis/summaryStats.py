import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set Seaborn style
sns.set(style="whitegrid")

# Load the data
df = pd.read_pickle('submission1/data/output/TaxBurden_Data.pkl')

# --- Task 1 ---
# Proportion of states with tax changes (1970-1985)
df_early = df[(df['Year'] >= 1970) & (df['Year'] <= 1985)]
df_early['tax_change'] = df_early.groupby('state')['tax_dollar'].diff().fillna(0) != 0
tax_change_per_year = df_early.groupby('Year')['tax_change'].mean() * 100  # to %

plt.figure(figsize=(12, 6))
sns.barplot(x=tax_change_per_year.index, y=tax_change_per_year.values, palette="Blues_d")
plt.title('Proportion of States with Cigarette Tax Change (1970-1985)', fontsize=14)
plt.ylabel('Proportion of States (%)')
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Task 2 ---
# Average tax (adjusted to 2012 dollars) and price per pack (1970-2018)
df['tax_2012_dollars'] = df['tax_dollar'] / df['price_cpi'] * 2.42  # CPI normalization
avg_by_year = df.groupby('Year').agg({'tax_2012_dollars': 'mean', 'cost_per_pack': 'mean'}).reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=avg_by_year, x='Year', y='tax_2012_dollars', label='Avg Tax (2012 $)', color='orange')
sns.lineplot(data=avg_by_year, x='Year', y='cost_per_pack', label='Avg Price per Pack', color='blue')
plt.title('Average Tax (2012 Dollars) and Price per Pack (1970-2018)', fontsize=14)
plt.ylabel('Dollars')
plt.xlabel('Year')
plt.legend()
plt.tight_layout()
plt.show()

# --- Task 3 ---
# Top 5 states by cigarette price increase and their sales per capita
price_change = df.groupby('state')['cost_per_pack'].agg(['first', 'last'])
price_change['increase'] = price_change['last'] - price_change['first']
top5_states = price_change.sort_values(by='increase', ascending=False).head(5).index.tolist()

df_top5 = df[df['state'].isin(top5_states)]
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_top5, x='Year', y='sales_per_capita', hue='state', palette='Set1')
plt.title('Packs Sold per Capita - Top 5 States with Highest Price Increases', fontsize=14)
plt.ylabel('Packs per Capita')
plt.xlabel('Year')
plt.legend(title='State')
plt.tight_layout()
plt.show()

# --- Task 4 ---
# Bottom 5 states by price increase and their sales per capita
bottom5_states = price_change.sort_values(by='increase').head(5).index.tolist()

df_bottom5 = df[df['state'].isin(bottom5_states)]
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_bottom5, x='Year', y='sales_per_capita', hue='state', palette='Set2')
plt.title('Packs Sold per Capita - Bottom 5 States with Lowest Price Increases', fontsize=14)
plt.ylabel('Packs per Capita')
plt.xlabel('Year')
plt.legend(title='State')
plt.tight_layout()
plt.show()

# --- Task 5 ---
# Comparison of trends in average sales (top vs bottom 5 states)
avg_sales_top = df_top5.groupby('Year')['sales_per_capita'].mean().reset_index()
avg_sales_bottom = df_bottom5.groupby('Year')['sales_per_capita'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=avg_sales_top, x='Year', y='sales_per_capita', label='Top 5 States', color='red')
sns.lineplot(data=avg_sales_bottom, x='Year', y='sales_per_capita', label='Bottom 5 States', color='green')
plt.title('Sales Trends: Top vs Bottom 5 Price Increase States', fontsize=14)
plt.ylabel('Average Packs Sold per Capita')
plt.xlabel('Year')
plt.legend()
plt.tight_layout()
plt.show()
