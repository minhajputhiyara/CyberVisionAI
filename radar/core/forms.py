from django import forms
from core.models import PcapFile, CsvFile

class PcapFileForm(forms.ModelForm):
    class Meta:
        model = PcapFile
        fields = ['name', 'pcap_file']

class CsvFileForm(forms.ModelForm):
    class Meta:
        model = CsvFile  # Use the CsvFile model
        fields = ['name', 'csv_file']  # Adjust fields as needed

class UploadCSVForm(forms.Form):
    csv_file = forms.FileField(label='Upload CSV File')
