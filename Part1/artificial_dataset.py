import csv
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate random data
num_rows = 300
num_cols = 6

# Generate mean and standard deviation for each column
means = np.random.uniform(low=0, high=10, size=num_cols)
std_devs = np.random.uniform(low=0.1, high=3, size=num_cols)

# Generate correlation matrix
correlation_matrix = np.random.uniform(low=-1, high=1, size=(num_cols, num_cols))
np.fill_diagonal(correlation_matrix, 1)  # Set diagonal to 1

# Ensure positive definiteness
correlation_matrix = np.dot(correlation_matrix, correlation_matrix.T)

# Generate data according to correlation matrix
data = np.random.multivariate_normal(means, np.diag(std_devs), size=num_rows)
data = np.dot(data, np.linalg.cholesky(correlation_matrix).T)

# Convert one column to integers
int_column_index = np.random.randint(0, num_cols)
print(int_column_index)
data[:, int_column_index] = data[:, int_column_index].astype(int)

# Save data to CSV
csv_file = "artificial_dataset.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([f"Column_{i}" for i in range(1, num_cols+1)])  # Write header
    writer.writerows(data)

print(f"CSV file '{csv_file}' has been generated successfully.")
