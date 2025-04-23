import os
import csv
import dpkt
import io

from django.http import HttpResponse
from django.conf import settings


from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import PcapFileForm, CsvFileForm
from .models import PcapFile, CsvFile
from django.shortcuts import render, get_object_or_404

import uuid  # Import UUID module
import csv
from scapy.all import rdpcap


from scapy.all import rdpcap, IP, TCP
from django.http import HttpResponse
import csv
import io


def analyse_pcap(request):
    if request.method == 'POST':
        # Get the primary key of the selected PcapFile instance
        pcap_file_id = request.POST.get('pcap_file_id')

        try:
            # Retrieve the PcapFile object using the primary key
            pcap_file = PcapFile.objects.get(pk=pcap_file_id)
        except PcapFile.DoesNotExist:
            return HttpResponse("Error: PcapFile not found.", status=404)

        pcap_file_path = pcap_file.pcap_file.path

        if os.path.isfile(pcap_file_path):
            # Load the pcap file
            packets = rdpcap(pcap_file_path)

            # Initialize CSV data list with headers
            csv_data = []
            csv_headers = ['No.', 'Time', 'Source', 'Destination', 'Info', 'Protocol', 'Length']
            csv_data.append(csv_headers)

            # Process the packets and extract desired fields
            for idx, packet in enumerate(packets):
                # Extract fields from the packet
                time = packet.time  # Packet arrival time
                source = packet.src if 'src' in packet.fields else 'N/A'  # Source IP address
                destination = packet.dst if 'dst' in packet.fields else 'N/A'  # Destination IP address
                info = packet.summary()  # Packet summary/info
                protocol = packet.name  # Protocol name
                length = len(packet)  # Packet length

                # Append extracted values to csv_data list
                csv_row = [idx + 1, time, source, destination, info, protocol, length]
                csv_data.append(csv_row)

            # Write CSV data to a string
            csv_string = ""
            if csv_data:
                output_file = io.StringIO()
                csv_writer = csv.writer(output_file)
                csv_writer.writerows(csv_data)
                csv_string = output_file.getvalue()

            # Return the CSV data as HTTP response
            response = HttpResponse(csv_string, content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="output.csv"'
            return response
        else:
            return HttpResponse("Error: PCAP file not found.", status=404)

    return HttpResponse("Method not allowed", status=405)




def upload_pcap_file(request):
    if request.method == 'POST':
        form = PcapFileForm(request.POST, request.FILES)
        if form.is_valid():
            # Generate a unique ID for pcap_id
            unique_id = str(uuid.uuid4())[:8]  # Generate a random UUID and take its first 8 characters
            # Get the uploaded file name and extension
            uploaded_file_name = request.FILES['pcap_file'].name
            file_extension = uploaded_file_name.split('.')[-1]
            # Combine unique_id with file extension to create pcap_id
            pcap_id = f"{unique_id}.{file_extension}"
            # Add pcap_id to form data and save
            instance = form.save(commit=False)
            instance.pcap_id = pcap_id
            instance.save()
            # Show success message
            messages.success(request, f"File '{uploaded_file_name}' uploaded successfully. PCAP ID: {pcap_id}")
            return redirect('upload_pcap_file')  # Redirect back to the upload page
    else:
        form = PcapFileForm()
    return render(request, 'core/upload_pcap.html', {'form': form})


def pcap_files(request):
    uploaded_files = PcapFile.objects.all()
    return render(request, 'core/pcap_files.html', {'uploaded_files': uploaded_files})

def csv_files(request):
    uploaded_files = CsvFile.objects.all()
    return render(request, 'core/csv_files.html', {'uploaded_files': uploaded_files})


def upload_csv_file(request):
    if request.method == 'POST':
        form = CsvFileForm(request.POST, request.FILES)
        if form.is_valid():
            # Generate a unique ID for pcap_id
            unique_id = str(uuid.uuid4())[:8]  # Generate a random UUID and take its first 8 characters
            # Get the uploaded file name and extension
            uploaded_file_name = request.FILES['csv_file'].name
            file_extension = uploaded_file_name.split('.')[-1]
            # Combine unique_id with file extension to create pcap_id
            csv_id = f"{unique_id}.{file_extension}"
            # Add pcap_id to form data and save
            instance = form.save(commit=False)
            instance.csv_id = csv_id
            instance.save()
            # Show success message
            messages.success(request, f"File '{uploaded_file_name}' uploaded successfully. CSV ID: {csv_id}")
            return redirect('csv_files')  # Redirect back to the upload page
    else:
        form = PcapFileForm()
    return render(request, 'core/upload_csv.html', {'form': form})




# def analysis_view(request):
#     if request.method == 'POST':
#         pcap_file_id = request.POST.get('pcap_file_id')

#         try:
#             # Retrieve the PcapFile object using the primary key
#             pcap_file = PcapFile.objects.get(pk=pcap_file_id)
#         except PcapFile.DoesNotExist:
#             return HttpResponse("Error: PcapFile not found.", status=404)

#         pcap_file_path = pcap_file.pcap_file.path

#         if os.path.isfile(pcap_file_path):
#             try:
#                 # Load the pcap file
#                 packets = rdpcap(pcap_file_path)

#                 # Define the output CSV file path
#                 output_csv_path = os.path.join(settings.MEDIA_ROOT, 'output.csv')

#                 # Define fieldnames for the CSV file
#                 fieldnames = [
#                     'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Protocol', 'Timestamp', 'Flow ID', 
#                     'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets', 'Total Length of Fwd Packet',
#                     'Total Length of Bwd Packet', 'Fwd Packet Length Max', 'Fwd Packet Length Min',
#                     'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max',
#                     'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
#                     'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
#                     'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
#                     'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
#                     'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
#                     'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Min', 'Packet Length Max',
#                     'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
#                     'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
#                     'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size',
#                     'Fwd Segment Size Avg', 'Bwd Segment Size Avg', 'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg',
#                     'Fwd Bulk Rate Avg', 'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg',
#                     'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
#                     'FWD Init Win Bytes', 'Bwd Init Win Bytes', 'Fwd Act Data Pkts', 'Fwd Seg Size Min',
#                     'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std',
#                     'Idle Max', 'Idle Min', 'Label', 'Label.1'
#                 ]

#                 # Write packet information to CSV
#                 with open(output_csv_path, 'w', newline='') as csvfile:
#                     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#                     writer.writeheader()

#                     for packet in packets:
#                         # Extract packet information
#                         # Modify the logic to extract the desired fields according to your requirement
#                         src_ip = packet[IP].src
#                         dst_ip = packet[IP].dst
#                         src_port = packet[TCP].sport if TCP in packet else None
#                         dst_port = packet[TCP].dport if TCP in packet else None
#                         protocol = packet[IP].proto
#                         timestamp = str(packet.time)

#                         # Write packet information to CSV
#                         writer.writerow({
#                             'Src IP': src_ip,
#                             'Dst IP': dst_ip,
#                             'Src Port': src_port,
#                             'Dst Port': dst_port,
#                             'Protocol': protocol,
#                             'Timestamp': timestamp,
#                             # Add more fields as needed
#                         })

#                 # Serve the generated CSV file for download
#                 with open(output_csv_path, 'rb') as f:
#                     response = HttpResponse(f, content_type='text/csv')
#                     response['Content-Disposition'] = 'attachment; filename=output.csv'
#                     return response
            
#             except Exception as e:
#                 return HttpResponse(f"Error converting pcap to CSV: {e}")

#         else:
#             return HttpResponse("Error: PCAP file not found.", status=404)

#     else:
#         return HttpResponse("Method not allowed", status=405)




def home_page(request):
    # Add any necessary logic here
    return render(request, 'core/home.html') 