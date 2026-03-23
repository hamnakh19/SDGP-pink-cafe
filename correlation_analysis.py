import pandas as pd
import numpy as np

# Load the cleaned data files
americano_df = pd.read_csv('cleaned_data/americano_cleaned.csv')
cappuccino_df = pd.read_csv('cleaned_data/cappuccino_cleaned.csv')
croissant_df = pd.read_csv('cleaned_data/croissants_cleaned.csv')

# Convert date to datetime
americano_df['date'] = pd.to_datetime(americano_df['date'])
cappuccino_df['date'] = pd.to_datetime(cappuccino_df['date'])
croissant_df['date'] = pd.to_datetime(croissant_df['date'])

# Rename columns to be more descriptive
americano_df = americano_df.rename(columns={'number_sold': 'Americano'})
cappuccino_df = cappuccino_df.rename(columns={'number_sold': 'Cappuccino'})
croissant_df = croissant_df.rename(columns={'number_sold': 'Croissant'})

# Merge all three datasets on date
df = americano_df.merge(cappuccino_df, on='date', how='outer')
df = df.merge(croissant_df, on='date', how='outer')
df = df.sort_values('date')

print("=== MERGED DATA (First 5 rows) ===")
print(df.head())
print("\n")

# Calculate correlation matrix
correlation_matrix = df[['Cappuccino', 'Americano', 'Croissant']].corr()

print("=== CORRELATION MATRIX ===")
print(correlation_matrix.round(3))
print("\n")

# Extract specific correlations
print("=== SPECIFIC CORRELATIONS ===")
print(f"Cappuccino & Americano: r = {correlation_matrix.loc['Cappuccino', 'Americano']:.3f}")
print(f"Croissant & Cappuccino: r = {correlation_matrix.loc['Croissant', 'Cappuccino']:.3f}")
print(f"Croissant & Americano: r = {correlation_matrix.loc['Croissant', 'Americano']:.3f}")

