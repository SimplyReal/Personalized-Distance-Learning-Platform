import pandas as pd
from sklearn.preprocessing import StandardScaler

# Read the CSV file
data = pd.read_csv('data.csv')

# Display the first few rows of the dataframe
print("Initial Data:")
print(data.head())

# Handle missing values (example: fill with mean)
data.fillna(data.mean(), inplace=True)

# Encode categorical variables (example: one-hot encoding)
data = pd.get_dummies(data)

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Convert the scaled data back to a dataframe
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

# Display the first few rows of the preprocessed dataframe
print("Preprocessed Data:")
print(data_scaled.head())

# Save the preprocessed data to a new CSV file
data_scaled.to_csv('preprocessed_data.csv', index=False)