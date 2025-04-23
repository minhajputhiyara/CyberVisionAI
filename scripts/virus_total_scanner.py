import pandas as pd
import requests
import os
import sys


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
    headers = {'x-apikey': api_key}

    try:
        # Send GET request to VirusTotal API
        response = requests.get(url, headers=headers)

        # Check if the request was successful (HTTP status code 200)
        response.raise_for_status()

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
            print("Invalid response from VirusTotal API.")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

    return None

def main(api_key, input_csv, output_csv, restart_file='restart_info.csv'):
    # Print name of file processed
    print(f"Processing file: {input_csv}")

    # Check if the output CSV file already exists
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        if existing_df.empty:
            # If the existing DataFrame is empty, initialize it with expected columns
            existing_df = pd.DataFrame(columns=['ip', 'as_owner', 'country', 'reputation', 'asn', 'malicious', 'suspicious', 'undetected', 'harmless', 'status'])
    else:
        # If the output CSV file doesn't exist, initialize the DataFrame with expected columns
        existing_df = pd.DataFrame(columns=['ip', 'as_owner', 'country', 'reputation', 'asn', 'malicious', 'suspicious', 'undetected', 'harmless', 'status'])

    # Check if the restart file exists
    restart_info = pd.read_csv(restart_file) if os.path.exists(restart_file) else pd.DataFrame()
    last_completed_ip = restart_info['last_completed_ip'].iloc[0] if not restart_info.empty else None

    # Read the input CSV file into a Pandas DataFrame
    df = pd.read_csv(input_csv)

    # Extract unique IP addresses from "ip" column
    unique_ips = df['ip'].dropna().unique()

    # Limit scanning to the first 500 unique IPs
    unique_ips = unique_ips[:500]

    # If a restart is needed, find the index of the last completed IP
    start_index = unique_ips.tolist().index(last_completed_ip) + 1 if last_completed_ip is not None else 0
    unique_ips = unique_ips[start_index:]

    total_ips = len(unique_ips)
    completed_ips = 0

    # Check each unique IP address against VirusTotal
    for ip in unique_ips:
        try:
            print(f"Scanning IP: {ip}")
            details = check_ip_malicious(api_key, ip)
            if details:
                # Create a DataFrame from the details and append it to the existing DataFrame
                result_df = pd.DataFrame([details])
                existing_df = pd.concat([existing_df, result_df], ignore_index=True)

            # Update completion percentage
            completed_ips += 1
            completion_percentage = (completed_ips / total_ips) * 100
            print(f"Completion Percentage: --------- {completion_percentage:.2f}%\r", end='', flush=True, file=sys.stdout)

        except requests.exceptions.RequestException as e:
            # Handle the error, possibly renewing the API key and retrying the failed IP
            print(f"Error: {e}")
            if "Quota exceeded" in str(e):
                print("API call limit exceeded.")
            # Save restart information
            restart_info = pd.DataFrame({'last_completed_ip': [ip]})
            restart_info.to_csv(restart_file, index=False)
            # Renew API key and retry
            # api_key = renew_api_key()
            # retry_failed_ip(api_key, ip, output_csv)

    # Save the combined DataFrame to the output CSV file
    existing_df.to_csv(output_csv, index=False, mode='w')
    print(f"\nResults saved to {output_csv}")

    # Remove the restart file as the process has completed
    if os.path.exists(restart_file):
        os.remove(restart_file)

if __name__ == "__main__":
    # Replace 'YOUR_API_KEY' with your actual VirusTotal API key
    api_key = '562ae93496fb3a73b6e374c4d642889604025ee15da6756c01a6f848698e5bbd'

    # Replace 'unique_ip_to_scan.csv' and 'output.csv' with your input and output file paths
    input_csv = 'unique_ip_to_scan.csv'
    output_csv = 'output.csv'

    main(api_key, input_csv, output_csv)


