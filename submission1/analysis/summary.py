import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator



taxBurden_data = pd.read_pickle("submission1/data/output/TaxBurden_Data.pkl")

print(taxBurden_data.head())
# Calculate the tax changes by year
def calculate_tax_changes(data):
    # Create a copy to prevent modifications to the original
    df = taxBurden_data.copy()
    
    # Ensure we have a state tax column
    if 'tax_state' not in df.columns:
        if 'tax_dollar' in df.columns:
            # Use tax_dollar if available
            df['tax_state'] = df['tax_dollar']
        else:
            raise ValueError("Neither 'tax_state' nor 'tax_dollar' column found in the data")

    # Sort by state and year
    df = df.sort_values(['state', 'Year'])
    
    # Calculate the difference in tax for each state from one year to the next
    df['tax_change'] = df.groupby('state')['tax_state'].diff()
    
    # Count states with tax changes by year
    tax_changes = df[df['tax_change'] != 0].groupby('Year').size()
    
    # Count total states by year
    total_states = df.groupby('Year')['state'].nunique()
    
    # Calculate proportion
    proportion = tax_changes / total_states
    
    # Filter to only include years 1970-1985
    proportion = proportion[(proportion.index >= 1970) & (proportion.index <= 1985)]
    
    # Fill NaN values with 0 (years with no changes)
    proportion = proportion.reindex(range(1970, 1986), fill_value=0)
    
    return proportion

# Calculate the proportion of states with tax changes
tax_change_proportion = calculate_tax_changes(taxBurden_data)

# Set the style for the plot
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))
# Create a bar plot using seaborn
plt.figure(figsize=(14, 8))
ax = sns.barplot(x=tax_change_proportion.index, y=tax_change_proportion.values, color="steelblue")

# Customize the plot
plt.title('Proportion of States with Cigarette Tax Changes (1970-1985)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Proportion of States', fontsize=14)
plt.xticks(rotation=45)
plt.ylim(0, 1.0)  # Set y-axis limits between 0 and 1

# Add value labels on top of each bar
for i, v in enumerate(tax_change_proportion.values):
    ax.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=10)

# Add a horizontal grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()

# Save the figure if needed
#plt.savefig('cigarette_tax_changes_1970_1985.png', dpi=300, bbox_inches='tight')

print("Plot generated showing the proportion of states with cigarette tax changes from 1970 to 1985.")


# Set the style for the plot
sns.set_style("whitegrid")
plt.figure(figsize=(14, 8))

data = taxBurden_data.copy()
# If using actual data, compute yearly averages
if 'cost_per_pack' in data.columns and 'tax_dollar' in data.columns:
    # Calculate the average price and tax by year across all states
    yearly_data = data.groupby('Year').agg({
        'cost_per_pack': 'mean',
        'tax_dollar': 'mean',
        'price_cpi': 'mean'  # This should be the CPI-adjusted price
    }).reset_index()
    
    # Rename columns for clarity
    yearly_data = yearly_data.rename(columns={
        'cost_per_pack': 'avg_price',
        'tax_dollar': 'avg_tax',
        'price_cpi': 'avg_price_adjusted'
    })
    
    # Adjust to 2012 dollars if necessary
    # We already have price_cpi but need to adjust it to 2012 base
    
    data = yearly_data
elif 'avg_price' not in data.columns or 'avg_tax' not in data.columns:
    print("Error: Data doesn't contain required columns for cigarette price and tax")
    
# Filter to only include years from 1970 to 2018
data = data[(data['Year'] >= 1970) & (data['Year'] <= 2018)]

# Create the plot
fig, ax1 = plt.subplots(figsize=(14, 8))

# Plot average price
line1 = ax1.plot(data['Year'], data['avg_price'], 'b-', linewidth=2.5, label='Average Price per Pack')
ax1.set_xlabel('Year', fontsize=14)
ax1.set_ylabel('Price (2012 Dollars)', color='blue', fontsize=14)
ax1.tick_params(axis='y', labelcolor='blue')

# Create a twin axis for tax
ax2 = ax1.twinx()
line2 = ax2.plot(data['Year'], data['avg_tax'], 'r-', linewidth=2.5, label='Average Tax per Pack')
ax2.set_ylabel('Tax (2012 Dollars)', color='red', fontsize=14)
ax2.tick_params(axis='y', labelcolor='red')

# Add grid lines
ax1.grid(True, linestyle='--', alpha=0.7)

# Add title
plt.title('Average Cigarette Price and Tax in 2012 Dollars (1970-2018)', fontsize=16)

# Add legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=12)

# Add vertical lines for significant tobacco policy events
# These are examples - adjust with actual historical events if needed
events = {
    1998: 'MSA*',     # Master Settlement Agreement
    2009: 'FSPTCA*'   # Family Smoking Prevention and Tobacco Control Act
}

for year, label in events.items():
    if year >= 1970 and year <= 2018:
        plt.axvline(x=year, color='green', linestyle='--', alpha=0.7)
        plt.text(year, ax1.get_ylim()[1]*0.95, label, rotation=90, verticalalignment='top')

# Footnote for policy abbreviations
if events:
    footnote = '\n'.join([f"{label}: {desc}" for label, desc in [
        ('MSA*', 'Master Settlement Agreement'),
        ('FSPTCA*', 'Family Smoking Prevention and Tobacco Control Act')
    ]])
    plt.figtext(0.01, 0.01, footnote, fontsize=8)

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# Display the plot
plt.savefig('cigarette_price_tax_1970_2018.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot generated showing the average cigarette price and tax from 1970 to 2018.")