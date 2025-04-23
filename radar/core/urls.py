from django.urls import path
from . import views
from .views import analyse_pcap
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home_page, name='home_page'),

    path('upload_pcap/', views.upload_pcap_file, name='upload_pcap_file'),
    path('upload_csv/', views.upload_csv_file, name='upload_csv'),

    path('pcap_files/', views.pcap_files, name='pcap_files'),
    path('csv_files/', views.csv_files, name='csv_files'),

    path('analyse/', analyse_pcap, name='analyse_pcap'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
