import pandas as pd
import requests

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
                    'id': attributes.get('id', ''),
                    'as_owner': attributes.get('as_owner', ''),
                    'country': attributes.get('country', ''),
                    'reputation': attributes.get('reputation', ''),
                    'asn': attributes.get('asn', ''),
                }

                if 'last_analysis_stats' in attributes:
                    analysis_stats = attributes['last_analysis_stats']
                    details['malicious'] = analysis_stats.get('malicious', 0)
                    details['suspicious'] = analysis_stats.get('suspicious', 0)
                    details['undetected'] = analysis_stats.get('undetected', 0)
                    details['harmless'] = analysis_stats.get('harmless', 0)

                    # Classify status based on analysis_stats
                    details['status'] = classify_status(analysis_stats)

                else:
                    details['malicious'] = details['suspicious'] = details['undetected'] = details['harmless'] = 0
                    details['status'] = 'unknown'

                return details

            else:
                print("Invalid response from VirusTotal API.")
        else:
            print(f"Error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"An error occurred: {e}")

    return None

def main(api_key, input_csv, output_csv):
    # Read the input CSV file into a Pandas DataFrame
    df = pd.read_csv(input_csv)

    # Create an empty list to store the results
    result_data = []

    # Extract unique IP addresses from the "ip" column
    unique_ips = df['ip'].dropna().unique()

    # Check each unique IP address against VirusTotal
    for ip in unique_ips:
        details = check_ip_malicious(api_key, ip)
        if details:
            result_data.append(details)

    # Create a DataFrame from the list of dictionaries
    result_df = pd.DataFrame(result_data)

    # Save the result DataFrame to a CSV file
    result_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    # Replace 'YOUR_API_KEY' with your actual VirusTotal API key
    api_key = 'f656f0e5f7d72aa71748bf189b3228ecba08c9736ae9274ec9a4ff1c03396220'

    # Replace 'input.csv' and 'output.csv' with your input and output file paths
    input_csv = './ds/output_chunk_1.csv'
    output_csv = 'fullcode_output.csv'

    main(api_key, input_csv, output_csv)

