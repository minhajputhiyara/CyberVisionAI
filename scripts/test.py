import pandas as pd
import requests
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Replace 'YOUR_API_KEY' with your actual VirusTotal API key
api_key = '1187f09e443134b298bf83ee7dc89d8938f5f22edd8a9ed81f5efb5d954aa074'

# Replace './ds/output_chunk_1.csv' and 'output.csv' with your input and output file paths
input_csv = '../ds/output_chunk_28.csv'
output_csv = 'output.csv'

def classify_status(analysis_stats):
    malicious_count = analysis_stats.get('malicious', 0)
    suspicious_count = analysis_stats.get('suspicious', 0)
    undetected_count = analysis_stats.get('undetected', 0)
    harmless_count = analysis_stats.get('harmless', 0)

    total_engines = malicious_count + suspicious_count + undetected_count + harmless_count

    # Define the threshold for classification
    malicious_threshold = 0.6
    harmless_threshold = 0.6

    # Classify status
    if malicious_count / total_engines > malicious_threshold:
        return 'malicious'
    elif harmless_count / total_engines > harmless_threshold:
        return 'benign'
    else:
        return 'unknown'

def check_ip_malicious(api_key, ip_address):
    # VirusTotal API endpoint for IP address reports
    url = f'https://www.virustotal.com/api/v3/ip_addresses/{ip_address}'

    # Set up headers with the API key
    headers = {
        'x-apikey': api_key
    }

    try:
        # Send GET request to VirusTotal API
        response = requests.get(url, headers=headers)
        
        # Check if the request was successful (HTTP status code 200)
        if response.status_code == 200:
            # Parse the JSON response
            result = response.json()

            # Check if the IP is flagged as malicious
            if 'data' in result and 'attributes' in result['data']:
                attributes = result['data']['attributes']

                # Extract relevant details
                details = {
                    'ip': ip_address,  # Use IP address as 'id'
                    'as_owner': attributes.get('as_owner', ''),
                    'country': attributes.get('country', ''),
                    'reputation': attributes.get('reputation', ''),
                    'asn': attributes.get('asn', ''),
                }

                if 'last_analysis_stats' in attributes:
                    analysis_stats = attributes['last_analysis_stats']
                    details.update({
                        'malicious': analysis_stats.get('malicious', 0),
                        'suspicious': analysis_stats.get('suspicious', 0),
                        'undetected': analysis_stats.get('undetected', 0),
                        'harmless': analysis_stats.get('harmless', 0),
                        'status': classify_status(analysis_stats),
                    })

                else:
                    details.update({
                        'malicious': 0,
                        'suspicious': 0,
                        'undetected': 0,
                        'harmless': 0,
                        'status': 'unknown',
                    })

                return details

            else:
                logging.error("Invalid response from VirusTotal API.")
        elif response.status_code == 429:
            logging.error(f"API limit exceeded. Last IP scanned: {ip_address}")
            sys.exit(1)
        else:
            logging.error(f"Error: {response.status_code} - {response.text}")

    except requests.exceptions.RequestException as e:
        logging.error(f"An error occurred: {e}")

    return None

def main(api_key, input_csv, output_csv):
    # Print the name of the file being processed
    logging.info(f"Processing file: {input_csv}")
    
    # Check if the output CSV file already exists
    if os.path.exists(output_csv):
        # Load the existing DataFrame
        existing_df = pd.read_csv(output_csv)
    else:
        # Create an empty DataFrame if the file doesn't exist
        existing_df = pd.DataFrame(columns=['ip', 'as_owner', 'country', 'reputation', 'asn', 'malicious', 'suspicious', 'undetected', 'harmless', 'status'])

    # Read the input CSV file into a Pandas DataFrame
    df = pd.read_csv(input_csv)

    # Extract unique IP addresses from the "ip" column
    unique_ips = df['ip'].dropna().unique()

    total_ips = len(unique_ips)
    completed_ips = 0

    # Create an empty list to store the results
    result_data = []

    # Check each unique IP address against VirusTotal
    for ip in unique_ips:
        sys.stdout.write(f"Scanning IP: {ip}\r")
        sys.stdout.flush()
        
        details = check_ip_malicious(api_key, ip)
        if details:
            result_data.append(details)

        # Update completion percentage
        completed_ips += 1
        completion_percentage = (completed_ips / total_ips) * 100
        sys.stdout.write(f"Completion Percentage : -------- {completion_percentage:.2f}%\r")
        sys.stdout.flush()

    # Create a DataFrame from the list of dictionaries
    result_df = pd.DataFrame(result_data)

    # Ensure consistent column order in both DataFrames
    column_order = ['ip', 'as_owner', 'country', 'reputation', 'asn', 'malicious', 'suspicious', 'undetected', 'harmless', 'status']
    existing_df = existing_df[column_order]
    result_df = result_df[column_order]

    # Combine the new results with the existing DataFrame
    combined_df = pd.concat([existing_df, result_df], ignore_index=True)

    # Save the combined DataFrame to the output CSV file
    try:
        combined_df.to_csv(output_csv, index=False)
        logging.info(f"\nResults appended to {output_csv}")
    except Exception as e:
        logging.error(f"Error saving data to {output_csv}: {e}")
        # Save the currently scanned data to output.csv in case of an exception
        result_df.to_csv(output_csv, index=False)
        logging.info(f"\nCurrent results saved to {output_csv}")

if __name__ == "__main__":
    # Replace './ds/output_chunk_1.csv' and 'output.csv' with your input and output file paths
    input_csv = '../ds/output_chunk_28.csv'
    output_csv = 'output.csv'

    try:
        main(api_key, input_csv, output_csv)
    except Exception as e:
        logging.error(f"An exception occurred: {e}")
        sys.exit(1)

