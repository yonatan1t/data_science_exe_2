import pandas as pd
import numpy as np

# Load Datasets
df = pd.read_csv('kickstarter_projects.csv')
new_df = pd.read_csv('new_projects.csv')

print("--- Training Data Overview ---")
print("Shape:", df.shape)
print("\nColumn Info:")
print(df.info())

print("\nMissing Values (Train):")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nTarget Distribution (state_ind):")
print(df.state_ind.value_counts(normalize=True))

print("\nTop Categories:")
print(df.category.value_counts(dropna=False).head(10))

print("\nCountry Distribution:")
print(df.country.value_counts(dropna=False))

print("\nStaff Pick Breakdown:")
print(df.staff_pick.value_counts(dropna=False))

print("\n--- Test Data Overview ---")
print("Shape:", new_df.shape)
print("Missing Values (Test):")
print(new_df.isnull().sum()[new_df.isnull().sum() > 0])
