from scapy.all import *
from scapy.layers import http

ttp_patterns = {
    "T1059.001": ["smtp"],
    "T1071": ["http"],
    "T1071.001": ["http"],
    "T1071.004": ["http"],
    "T1071.005": ["dns"],
    # Add more TTPs and associated patterns as needed
}

def extract_data_from_packet(packet):
    try:
        time = packet.time
        src = packet.src
        dst = packet.dst
        proto = packet.getlayer(IP).proto
        length = len(packet)
        info = getattr(packet, 'info', '')
        return time, src, dst, proto, length, info
    except Exception as e:
        raise ValueError(f"Error extracting data from packet: {e}")

def extract_mail_data(packet):
    try:
        if packet.haslayer(TCP) and packet[TCP].dport == 25:
            return packet.load
    except Exception as e:
        raise ValueError(f"Error extracting mail data from packet: {e}")

def extract_http_data(packet):
    try:
        if packet.haslayer(http.HTTPRequest):
            return str(packet[http.HTTPRequest].payload)
    except Exception as e:
        raise ValueError(f"Error extracting HTTP data from packet: {e}")

def extract_dns_data(packet):
    try:
        if packet.haslayer(DNS):
            return packet[DNS].summary()
    except Exception as e:
        raise ValueError(f"Error extracting DNS data from packet: {e}")

def match_ttps(packet):
    try:
        proto = packet.getlayer(IP).proto
        matched_ttps = [ttp for ttp, patterns in ttp_patterns.items() if any(pattern in proto.lower() for pattern in patterns)]
        return matched_ttps
    except Exception as e:
        raise ValueError(f"Error matching TTPs: {e}")

def main():
    try:
        pcap_file = "000A1C469454184BBCDE37F868818EF7A3E5EEFF0E70D0C5F1E4C791DE2C2F30.pcap"
        packets = rdpcap(pcap_file)

        for packet in packets:
            time, src, dst, proto, length, info = extract_data_from_packet(packet)
            mail_data = extract_mail_data(packet)
            http_data = extract_http_data(packet)
            dns_data = extract_dns_data(packet)

            matched_ttps = match_ttps(packet)
            print(f"Matched TTPs: {matched_ttps}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
