import pandas as pd


df = pd.read_csv('app\data\creditcard_sampled.csv')

# Create a new 'Day' column (1st day is 0)
df['Hour'] = (df['Time'] // 3600).astype(int)

total_per_hour = df.groupby("Hour").size()


fraud_per_hour = df[df['Class'] == 1].groupby('Hour').size()

hourly_stats = pd.DataFrame({
    'Total': total_per_hour,
    'Fraud': fraud_per_hour
}).fillna(0).astype(int)

# For the chart
labels = [f"Hour {h}" for h in hourly_stats.index]
fraud_values = hourly_stats['Fraud'].tolist()
total_values = hourly_stats['Total'].tolist()

print("Labels:", labels)
print("Fraud values:", fraud_values)
print("Total values:", total_values)
