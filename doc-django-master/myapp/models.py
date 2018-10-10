from django.db import models
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import os

class OverwriteStorage(FileSystemStorage):
    def get_available_name(self, name):
        if self.exists(name):
            os.remove(os.path.join(settings.MEDIA_ROOT, name))
        return name
    
class Document(models.Model):
    myfile = "./media/documents/test_data.txt"
    if os.path.isfile(myfile):
        os.remove(myfile)
    else:    ## Show an error ##
        print("%s file not found1" % myfile)    
    docfile = models.FileField(upload_to='documents')
   # docfile.save()
    #models.file.delete()
    #models.file.save()
    #media = models.FileField(u"Arquivo", upload_to=settings.MEDIA_DIR, storage=OverwriteStorage())