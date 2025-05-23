# Generated by Django 5.0.3 on 2024-04-07 19:23

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='CsvFile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('csv_id', models.CharField(max_length=100)),
                ('name', models.CharField(max_length=100)),
                ('csv_file', models.FileField(upload_to='csv_files/')),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
                ('status', models.CharField(choices=[('Not analysed', 'Not analysed'), ('Processing', 'Processing'), ('Analysed', 'Analysed')], default='Not analysed', max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='IPScanResult',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ip', models.CharField(max_length=50)),
                ('pcap_file', models.CharField(max_length=100)),
                ('as_owner', models.CharField(blank=True, max_length=200, null=True)),
                ('country', models.CharField(blank=True, max_length=50, null=True)),
                ('reputation', models.CharField(blank=True, max_length=50, null=True)),
                ('asn', models.CharField(blank=True, max_length=50, null=True)),
                ('malicious', models.IntegerField(default=0)),
                ('suspicious', models.IntegerField(default=0)),
                ('undetected', models.IntegerField(default=0)),
                ('harmless', models.IntegerField(default=0)),
                ('status', models.CharField(default='unknown', max_length=50)),
            ],
        ),
        migrations.CreateModel(
            name='PcapFile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('pcap_id', models.CharField(max_length=100)),
                ('name', models.CharField(max_length=100)),
                ('pcap_file', models.FileField(upload_to='pcap_files/')),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
                ('status', models.CharField(choices=[('Not analysed', 'Not analysed'), ('Processing', 'Processing'), ('Analysed', 'Analysed')], default='Not analysed', max_length=100)),
            ],
        ),
    ]
