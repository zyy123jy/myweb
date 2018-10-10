import csv

from django.shortcuts import render

from django.template import RequestContext
from django.http import HttpResponseRedirect
from django.http import HttpResponse
from django.urls import reverse

from myapp.models import Document
from myapp.forms import DocumentForm

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
import os
import datetime
import random
from . import compute_model as cm

def show(request):
    # Handle file upload   
    myfile = "/home/zyy/Documents/kdd-master/myweb/doc-django-master/media/documents/test_data.txt"
    form = DocumentForm() # A empty, unbound form
    
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile = request.FILES['docfile'])
            newdoc.save()
          #  if os.path.isfile(myfile):
            #plot(request)
        return HttpResponseRedirect(reverse('show'))
    
    data = process_document(myfile)
    text = " | ".join([", ".join(row) for row in data])
    return render(request, 'show.html', {'text': text, 'form': form, 'graphic': ""})
     

def process_document(myfile):
    data = []
    if myfile != None and os.path.isfile(myfile):
        reader = csv.reader(open(myfile,'r'))
        for row in reader:
            data.append(row)

    return data


def plot(request):
    # dummy data
    T_test,T_pre,std_mmse,mean_mmse,std_adas,mean_adas,std_cdr,mean_cdr,time_scale,base_age,Me,Y1_test,Y2_test,Y5_test,mmse_scale,adas_scale,cd_scale = cm.train_model()
    fig = Figure()
    ax=fig.add_subplot(3,1,1)
    ax.plot(T_test[0,0:Me]*time_scale+base_age,Y1_test[0,0:Me]*mmse_scale,'ro')
    ax.plot(T_pre[0,:]*time_scale+base_age, mean_mmse+std_mmse,'r--',linewidth=1)
    ax.plot(T_pre[0,:]*time_scale+base_age, mean_mmse,'r',linewidth=4)
    ax.plot(T_pre[0,:]*time_scale+base_age, mean_mmse-std_mmse,'r--',linewidth=1)
    ax.fill_between(T_pre[0,:]*time_scale+base_age, mean_mmse-std_mmse, mean_mmse+std_mmse, color='r', alpha=0.2)
    ax.set_xlabel('Age(Years)')
    ax.set_ylabel('MMSE')
    print('plot 1')  
    ax=fig.add_subplot(3,1,2)
    ax.plot(T_test[0,0:Me]*time_scale+base_age,Y2_test[0,0:Me]*adas_scale,'go')
    ax.plot(T_pre[0,:]*time_scale+base_age, mean_adas+std_adas,'g--',linewidth=1)
    ax.plot(T_pre[0,:]*time_scale+base_age, mean_adas,'g',linewidth=4)
    ax.plot(T_pre[0,:]*time_scale+base_age, mean_adas-std_adas,'g--',linewidth=1)
    ax.fill_between(T_pre[0,:]*time_scale+base_age, mean_adas-std_adas, mean_adas+std_adas, color='g', alpha=0.2)
    ax.set_xlabel('Age(Years)')
    ax.set_ylabel('ADAS-COG')
    print('plot 2')
   
    ax=fig.add_subplot(3,1,3)
    ax.plot(T_test[0,0:Me]*time_scale+base_age,Y5_test[0,0:Me]*cd_scale,'bo')
    ax.plot(T_pre[0,:]*time_scale+base_age, mean_cdr+std_cdr,'b--',linewidth=1)
    ax.plot(T_pre[0,:]*time_scale+base_age, mean_cdr,'b',linewidth=4)
    ax.plot(T_pre[0,:]*time_scale+base_age, mean_cdr-std_cdr,'b--',linewidth=1)
    ax.fill_between(T_pre[0,:]*time_scale+base_age, mean_cdr-std_cdr, mean_cdr+std_cdr, color='b', alpha=0.2)
    ax.set_xlabel('Age(Years)')
    ax.set_ylabel('CDR-SB')
    canvas = FigureCanvas(fig)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)
    print('plot 3')
    myfile = "/home/zyy/Documents/kdd-master/myweb/doc-django-master/media/documents/test_data.txt"
    if os.path.isfile(myfile):
        os.remove(myfile)
    else:    ## Show an error ##
        print("%s file not found1" % myfile)   
        
    return response
