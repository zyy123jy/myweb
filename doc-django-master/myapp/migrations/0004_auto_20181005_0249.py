# Generated by Django 2.1.1 on 2018-10-05 02:49

from django.db import migrations, models
import myapp.models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0003_auto_20181005_0237'),
    ]

    operations = [
        migrations.AlterField(
            model_name='document',
            name='docfile',
            field=models.FileField(storage=myapp.models.OverwriteStorage(), upload_to='documents'),
        ),
    ]
