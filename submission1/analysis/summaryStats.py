import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
taxBurden_data = pd.read_pickle("submission1/data/output/TaxBurden_Data.pkl")

# Step 1: Calculate price increases for each state over the period 1970-1985
def calculate_price_changes():
    # Create a copy to prevent modifications to the original
    df = taxBurden_data.copy()
    
    # Filter for years 1970 and 1985
    df_filtered = df[(df['Year'] == 1970) | (df['Year'] == 1985)]
    
    # Create a pivot table to get the first and last year prices for each state
    pivot_df = df_filtered.pivot_table(index='state', columns='Year', values='cost_per_pack')
    
    # Calculate the price increase (in dollars)
    pivot_df['price_increase'] = pivot_df[1985] - pivot_df[1970]
    
    # Sort by price increase and get the 5 states with highest increases
    highest_increase = pivot_df.sort_values('price_increase', ascending=False).head(5)
    highest_states = highest_increase.index.tolist()
    
    # Get the 5 states with lowest increases
    lowest_increase = pivot_df.sort_values('price_increase').head(5)
    lowest_states = lowest_increase.index.tolist()
    
    return highest_states, lowest_states, pivot_df

# Step 2: Plot packs sold per capita for the identified states from 1970 to 2018
def plot_packs_per_capita(states_list, title_suffix):
    # Create a copy to prevent modifications to the original
    df = taxBurden_data.copy()
    
    # Filter for the specified states and years between 1970 and 2018
    df_filtered = df[(df['state'].isin(states_list)) & 
                     (df['Year'] >= 1970) & 
                     (df['Year'] <= 2018)]
    
    # Plot setup
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    # Create the line plot
    ax = sns.lineplot(x='Year', y='sales_per_capita', hue='state', data=df_filtered, linewidth=2.5)
    
    # Customize the plot
    plt.title(f'Average Number of Packs Sold Per Capita (1970-2018): {title_suffix}', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Packs Per Capita', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='State', title_fontsize=12, fontsize=10, loc='best')
    
    # Add vertical lines for significant tobacco policy events
    events = {
        1998: 'MSA',     # Master Settlement Agreement
        2009: 'FSPTCA'   # Family Smoking Prevention and Tobacco Control Act
    }
    
    for year, label in events.items():
        plt.axvline(x=year, color='green', linestyle='--', alpha=0.7)
        plt.text(year, ax.get_ylim()[1]*0.95, label, rotation=90, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(f'packs_per_capita_{title_suffix.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df_filtered

# Step 3: Compare trends between highest and lowest price increase states
def compare_trends(highest_df, lowest_df):
    # Create a copy of the dataframes and add a group column
    highest_df = highest_df.copy()
    highest_df['price_group'] = 'Highest Increase States'
    
    lowest_df = lowest_df.copy()
    lowest_df['price_group'] = 'Lowest Increase States'
    
    # Combine the dataframes
    combined_df = pd.concat([highest_df, lowest_df])
    
    # Calculate yearly averages by group
    yearly_avg = combined_df.groupby(['Year', 'price_group'])['sales_per_capita'].mean().reset_index()
    
    # Plot setup
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    # Create the line plot
    ax = sns.lineplot(x='Year', y='sales_per_capita', hue='price_group', 
                  data=yearly_avg, linewidth=3.5)
    
    # Customize the plot
    plt.title('Comparison of Average Packs Sold Per Capita: Highest vs. Lowest Price Increase States (1970-2018)', 
              fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Average Packs Per Capita', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='State Group', title_fontsize=12, fontsize=10, loc='best')
    
    # Add vertical lines for significant tobacco policy events
    events = {
        1998: 'MSA',     # Master Settlement Agreement
        2009: 'FSPTCA'   # Family Smoking Prevention and Tobacco Control Act
    }
    
    for year, label in events.items():
        plt.axvline(x=year, color='green', linestyle='--', alpha=0.7)
        plt.text(year, ax.get_ylim()[1]*0.95, label, rotation=90, verticalalignment='top')
    
    # Add annotations explaining the abbreviations
    plt.figtext(0.01, 0.01, "MSA: Master Settlement Agreement\nFSPTCA: Family Smoking Prevention and Tobacco Control Act", 
                fontsize=8)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('comparison_highest_lowest_states.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate percentage decrease in sales from 1970 to 2018 for each group
    start_year = 1970
    end_year = 2018
    
    highest_start = yearly_avg[(yearly_avg['Year'] == start_year) & 
                              (yearly_avg['price_group'] == 'Highest Increase States')]['sales_per_capita'].values[0]
    highest_end = yearly_avg[(yearly_avg['Year'] == end_year) & 
                            (yearly_avg['price_group'] == 'Highest Increase States')]['sales_per_capita'].values[0]
    highest_pct_change = ((highest_end - highest_start) / highest_start) * 100
    
    lowest_start = yearly_avg[(yearly_avg['Year'] == start_year) & 
                             (yearly_avg['price_group'] == 'Lowest Increase States')]['sales_per_capita'].values[0]
    lowest_end = yearly_avg[(yearly_avg['Year'] == end_year) & 
                           (yearly_avg['price_group'] == 'Lowest Increase States')]['sales_per_capita'].values[0]
    lowest_pct_change = ((lowest_end - lowest_start) / lowest_start) * 100
    
    print(f"Percentage change in sales per capita (1970-2018):")
    print(f"Highest price increase states: {highest_pct_change:.2f}%")
    print(f"Lowest price increase states: {lowest_pct_change:.2f}%")
    
    # Calculate price elasticity estimation
    price_increase_highest = pivot_df.loc[highest_states, 'price_increase'].mean()
    price_increase_lowest = pivot_df.loc[lowest_states, 'price_increase'].mean()
    
    print(f"\nAverage price increase (1970-1985):")
    print(f"Highest price increase states: ${price_increase_highest:.2f}")
    print(f"Lowest price increase states: ${price_increase_lowest:.2f}")

# Main execution
print("Calculating price changes and identifying states (1970-1985)...")
highest_states, lowest_states, pivot_df = calculate_price_changes()

print("\nStates with highest cigarette price increases (1970-1985):")
for state in highest_states:
    increase = pivot_df.loc[state, 'price_increase']
    print(f"{state}: ${increase:.2f}")

print("\nStates with lowest cigarette price increases (1970-1985):")
for state in lowest_states:
    increase = pivot_df.loc[state, 'price_increase']
    print(f"{state}: ${increase:.2f}")

# Question 3: Plot for highest price increase states
print("\nCreating plot for states with highest price increases...")
highest_df = plot_packs_per_capita(highest_states, "States with Highest Price Increases (1970-1985)")

# Question 4: Plot for lowest price increase states
print("\nCreating plot for states with lowest price increases...")
lowest_df = plot_packs_per_capita(lowest_states, "States with Lowest Price Increases (1970-1985)")

# Question 5: Compare trends between the two groups
print("\nComparing trends between highest and lowest price increase states...")
compare_trends(highest_df, lowest_df)