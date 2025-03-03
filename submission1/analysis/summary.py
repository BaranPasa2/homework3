import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns



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