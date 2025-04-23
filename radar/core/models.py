from django.db import models

class PcapFile(models.Model):
    STATUS_CHOICES = (
        ('Not analysed', 'Not analysed'),
        ('Processing', 'Processing'),
        ('Analysed', 'Analysed'),
    )

    pcap_id = models.CharField(max_length=100)
    name = models.CharField(max_length=100)
    pcap_file = models.FileField(upload_to='pcap_files/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=100, choices=STATUS_CHOICES, default='Not analysed')

    def __str__(self):
        return self.name

class CsvFile(models.Model):
    STATUS_CHOICES = (
        ('Not analysed', 'Not analysed'),
        ('Processing', 'Processing'),
        ('Analysed', 'Analysed'),
    )

    csv_id = models.CharField(max_length=100)
    name = models.CharField(max_length=100)
    csv_file = models.FileField(upload_to='csv_files/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=100, choices=STATUS_CHOICES, default='Not analysed')

    def __str__(self):
        return self.name
    
class IPScanResult(models.Model):
    ip = models.CharField(max_length=50)
    pcap_file = models.CharField(max_length=100)  # New field to identify the pcap file
    as_owner = models.CharField(max_length=200, blank=True, null=True)
    country = models.CharField(max_length=50, blank=True, null=True)
    reputation = models.CharField(max_length=50, blank=True, null=True)
    asn = models.CharField(max_length=50, blank=True, null=True)
    malicious = models.IntegerField(default=0)
    suspicious = models.IntegerField(default=0)
    undetected = models.IntegerField(default=0)
    harmless = models.IntegerField(default=0)
    status = models.CharField(max_length=50, default='unknown')

    def __str__(self):
        return f"{self.ip} - {self.pcap_file}"