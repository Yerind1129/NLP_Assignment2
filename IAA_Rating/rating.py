import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

# Load data
df = pd.read_csv('scoring.csv')  # Replace with your CSV path

# Categories to evaluate
categories = ['correctness', 'completeness', 'relevance', 'clarity', 'hallucination']

# Store results
results = []

# Calculate IAA for each category
for category in categories:
    # Get columns for this category
    col1 = f"{category}_1"
    col2 = f"{category}_2"
    
    if col1 in df.columns and col2 in df.columns:
        # Extract ratings and drop rows with NaN values
        valid_rows = df[[col1, col2]].dropna()
        ratings1 = valid_rows[col1].values
        ratings2 = valid_rows[col2].values
        
        # Skip if no valid rows are left
        if len(ratings1) == 0:
            print(f"Skipping {category} due to no valid rows after dropping NaN values.")
            continue
        
        # Calculate exact agreement percentage
        exact_match = np.mean(ratings1 == ratings2) * 100
        
        # Calculate Cohen's Kappa
        kappa = cohen_kappa_score(ratings1, ratings2, weights='linear')
        
        # Calculate within-1 agreement
        within_one = np.mean(np.abs(ratings1 - ratings2) <= 1) * 100
        
        results.append({
            'Category': category,
            'Exact Agreement (%)': round(exact_match, 2),
            'Weighted Kappa': round(kappa, 4),
            'Within-1 Agreement (%)': round(within_one, 2)
        })

# Calculate overall agreement if available
if 'overall_1' in df.columns and 'overall_2' in df.columns:
    # Extract ratings and drop rows with NaN values
    valid_rows = df[['overall_1', 'overall_2']].dropna()
    overall1 = valid_rows['overall_1'].values
    overall2 = valid_rows['overall_2'].values
    
    # Skip if no valid rows are left
    if len(overall1) > 0:
        exact_match = np.mean(overall1 == overall2) * 100
        kappa = cohen_kappa_score(overall1, overall2, weights='linear')
        within_one = np.mean(np.abs(overall1 - overall2) <= 1) * 100
        
        results.append({
            'Category': 'overall',
            'Exact Agreement (%)': round(exact_match, 2),
            'Weighted Kappa': round(kappa, 4),
            'Within-1 Agreement (%)': round(within_one, 2)
        })

# Convert to DataFrame for nicer display
results_df = pd.DataFrame(results)

# Interpret Kappa values
def interpret_kappa(k):
    if k < 0:
        return "Poor"
    elif k < 0.2:
        return "Slight"
    elif k < 0.4:
        return "Fair"
    elif k < 0.6:
        return "Moderate"
    elif k < 0.8:
        return "Substantial"
    else:
        return "Almost Perfect"

results_df['Agreement Level'] = results_df['Weighted Kappa'].apply(interpret_kappa)

# Display results
print(results_df)

# Save results
results_df.to_csv('iaa_results.csv', index=False)
print("\nResults saved to 'iaa_results.csv'")

# Calculate average IAA across all categories
avg_kappa = results_df['Weighted Kappa'].mean()
avg_exact = results_df['Exact Agreement (%)'].mean()

print(f"\nOverall statistics:")
print(f"- Average Weighted Kappa: {avg_kappa:.4f} ({interpret_kappa(avg_kappa)})")
print(f"- Average Exact Agreement: {avg_exact:.2f}%")