import pandas as pd

def break_csv(input_csv, max_rows_per_file=200):
    # Read the input CSV file into a Pandas DataFrame
    df = pd.read_csv(input_csv)

    # Get the total number of rows in the DataFrame
    total_rows = df.shape[0]

    # Calculate the number of files needed
    num_files = (total_rows + max_rows_per_file - 1) // max_rows_per_file

    # Split the DataFrame into chunks of max_rows_per_file
    chunks = [df[i * max_rows_per_file:(i + 1) * max_rows_per_file] for i in range(num_files)]

    # Save each chunk to a separate CSV file
    for i, chunk in enumerate(chunks):
        output_csv = f'output_chunk_{i + 1}.csv'
        chunk.to_csv(output_csv, index=False)
        print(f"Chunk {i + 1} saved to {output_csv}")

if __name__ == "__main__":
    # Replace 'input.csv' with your actual input file path
    input_csv = 'unique_ip_to_scan.csv'

    # Specify the maximum number of rows per file (default is 500)
    max_rows_per_file = 200

    # Call the function to break the CSV file
    break_csv(input_csv, max_rows_per_file)
