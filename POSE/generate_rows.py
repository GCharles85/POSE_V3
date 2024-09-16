import pandas as pd
import random
import sys

def generate_random_subset_excel(num_rows, output_file="output.xlsx"):
    """
    Generates an Excel file with a specified number of rows, 
    where each row contains a random subset of a hardcoded car issues list.
    
    Args:
    - num_rows (int): Number of rows to generate.
    - output_file (str): Path to the output Excel file (default is 'output.xlsx').
    """
    # Hardcoded list of car issues
    car_issues = ['Engine Overheating', 'Brake Failure', 'Transmission Issue', 'Flat Tire', 'Battery Dead']
    
    # Create an empty list to hold the rows of random subsets
    data = []
    
    for _ in range(num_rows):
        # Generate a random subset of the car issues list (at least one issue)
        subset = random.sample(car_issues, random.randint(1, len(car_issues)))
        # Add the subset to the data list
        data.append(subset)
    
    # Convert the list of subsets into a DataFrame (uneven rows will be filled with NaN)
    df = pd.DataFrame(data)
    
    # Output the DataFrame to an Excel file
    df.to_excel(output_file, index=False)
    
    print(f"Excel file with {num_rows} rows saved as '{output_file}'.")

if __name__ == "__main__":
    # Check if the correct number of arguments is passed
    if len(sys.argv) != 2:
        print("Usage: python script.py <num_rows>")
        sys.exit(1)
    
    # Get the number of rows from the command line argument
    try:
        num_rows = int(sys.argv[1])
    except ValueError:
        print("Please provide a valid integer for the number of rows.")
        sys.exit(1)
    
    # Call the function to generate the Excel file
    generate_random_subset_excel(num_rows)
