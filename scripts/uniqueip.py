import pandas as pd

def count_unique_ips(csv_file):
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_file)

    # Extract unique IP addresses from "Src IP" and "Dst IP" columns
    src_ips = df['Src IP'].dropna().unique()
    dst_ips = df['Dst IP'].dropna().unique()

    # Combine and remove duplicates
    unique_ips = set(src_ips) | set(dst_ips)

    # Print the total number of non-duplicate IP addresses
    print(f"Total number of non-duplicate IP addresses: {len(unique_ips)}")

if __name__ == "__main__":
    # Replace 'input.csv' with your actual input file path
    input_csv = 'input.csv'

    count_unique_ips(input_csv)
