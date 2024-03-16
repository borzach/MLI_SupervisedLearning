import numpy as np

def read_npy_file(file_path, num_lines=5):
    try:
        # Load the .npy file
        data = np.load(file_path, allow_pickle=True)
        
        # Extract column names and data
        column_names = data[0]
        data = data[1:]
        
        # Print the column names
        print("Column Names:")
        print(column_names)
        
        # Print the contents of the file
        print("\nContents of the npy file:")
        print(data)
        
        return column_names, data
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
file_path = "X_test.npy"  # Replace this with the path to your .npy file
data = read_npy_file(file_path)
