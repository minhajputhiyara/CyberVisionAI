from django.contrib import admin
from .models import PcapFile, CsvFile
# Register your models here.


admin.site.register(PcapFile)
admin.site.register(CsvFile)