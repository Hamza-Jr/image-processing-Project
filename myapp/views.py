from django.shortcuts import render, HttpResponse
import os
from django.conf import settings
import uuid
from .noises import *
from .filtrage import *
from.traitmantparpoint import * 
from.filtrage_frequentiel import * 
from.countour_detection import *
from.morphology import *  
from.filtrage_spatial import *

# Create your views here.

def home(request):
    return render(request, 'index.html')

def upload(request):
    if request.method == "POST":
        image_file = request.FILES['image']
        file_extension = os.path.splitext(image_file.name)[1]
        if file_extension == '.png' or file_extension == '.jpeg' or file_extension == '.jpg':
            request.session['id'] = str(uuid.uuid4())
            request.session['image_in_path'] = f"images/{request.session['id']}_input{file_extension}"
            print(request.session['image_in_path'])
            request.session['image_noise_path'] = request.session['image_in_path']
            request.session['image_out_path'] = request.session['image_in_path']
            with open(os.path.join(settings.STATIC_ROOT, request.session['image_in_path']), 'wb') as f:
                f.write(image_file.read())
        else:
            alert = True
            return render(request,'index.html',{'alert':alert})
    return render(request, "index.html", {'image_out_path': request.session['image_out_path'], "image_noise_path": request.session['image_noise_path']})
    
    
def applyNoiseGaussien(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_in_path']))
        n = add_gaussian_noise(image)
        if(request.session['image_noise_path'] == request.session['image_in_path']):
            request.session['image_noise_path'] = f"images/{request.session['id']}_noise.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
    s1 = "checked"
    s2 = ""
    s3 = ""
    s4 = ""
    s5 = ""
    s6 = ""
    s7 = ""
    s8 = ""
    s9= ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "s1": s1, "s2": s2, "s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6, "s7": s7, "s8": s8, "s9": s9})


def applyNoiseSaltPepper(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_in_path']))
        n = add_salt_and_pepper_noise(image)
        if(request.session['image_noise_path'] == request.session['image_in_path']):
            request.session['image_noise_path'] = f"images/{request.session['id']}_noise.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
    s1 = ""
    s2 = "checked"
    s3 = ""
    s4 = ""
    s5 = ""
    s6 = ""
    s7 = ""
    s8 = ""
    s9= ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6, "s7": s7, "s8": s8, "s9": s9})


def applyNoisePeriodic(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_in_path']))
        n = add_periodic_noise(image)
        if(request.session['image_noise_path'] == request.session['image_in_path']):
            request.session['image_noise_path'] = f"images/{request.session['id']}_noise.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
    s1 = ""
    s2 = ""
    s3 = ""
    s4 = ""
    s5 = ""
    s6 = ""
    s7 = "checked"
    s8 = ""
    s9= ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6, "s7": s7, "s8": s8, "s9": s9})

def applyNoiseQuantisation(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_in_path']))
        n = add_quantization_noise(image)
        if(request.session['image_noise_path'] == request.session['image_in_path']):
            request.session['image_noise_path'] = f"images/{request.session['id']}_noise.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
    s1 = ""
    s2 = ""
    s3 = ""
    s4 = ""
    s5 = ""
    s6 = "checked"
    s7 = ""
    s8 = ""
    s9= ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6, "s7": s7, "s8": s8, "s9": s9})

def applyNoisePoisson(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_in_path']))
        n = add_poisson_noise(image)
        if(request.session['image_noise_path'] == request.session['image_in_path']):
            request.session['image_noise_path'] = f"images/{request.session['id']}_noise.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
    s1 = ""
    s2 = ""
    s3 = "checked"
    s4 = ""
    s5 = ""
    s6 = ""
    s7 = ""
    s8 = ""
    s9= ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6, "s7": s7, "s8": s8, "s9": s9})

def applyNoiseSpeckle(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_in_path']))
        n = add_speckle_noise(image)
        if(request.session['image_noise_path'] == request.session['image_in_path']):
            request.session['image_noise_path'] = f"images/{request.session['id']}_noise.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
    s1 = ""
    s2 = ""
    s3 = ""
    s4 = "checked"
    s5 = ""
    s6 = ""
    s7 = ""
    s8 = ""
    s9= ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6, "s7": s7, "s8": s8, "s9": s9})


def applyNoiseImpulse(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_in_path']))
        n = add_impulse_noise(image)
        if(request.session['image_noise_path'] == request.session['image_in_path']):
            request.session['image_noise_path'] = f"images/{request.session['id']}_noise.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
    s1 = ""
    s2 = ""
    s3 = ""
    s4 = ""
    s5 = "checked"
    s6 = ""
    s7 = ""
    s8 = ""
    s9= ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6, "s7": s7, "s8": s8, "s9": s9})

def applyimage_original(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_in_path']))
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if(request.session['image_noise_path'] == request.session['image_in_path']):
            request.session['image_noise_path'] = f"images/{request.session['id']}_noise.jpg"
        save_image_data(rgb_image, os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        request.session['image_noise_path'] == request.session['image_in_path']
    s1 = ""
    s2 = ""
    s3 = ""
    s4 = ""
    s5 = ""
    s6 = ""
    s7 = ""
    s8 = "checked"
    s9= ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6, "s7": s7, "s8": s8, "s9": s9})

def applyimage_original_grise(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_in_path']))
        n = image_original_grise(image)
        if(request.session['image_noise_path'] == request.session['image_in_path']):
            request.session['image_noise_path'] = f"images/{request.session['id']}_noise.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
    s1 = ""
    s2 = ""
    s3 = ""
    s4 = ""
    s5 = ""
    s6 = ""
    s7 = ""
    s8 = ""
    s9= "checked"
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6, "s7": s7, "s8": s8, "s9": s9})


#Traitment par point
def applyTransExp(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = exponential_transformation(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "block"
    t2 = "none"
    t3 = "none"
    t4 = "none"
    t5 = "none"
    tpp1 = "checked"
    tpp2 = ""
    tpp3 = ""
    tpp4 = ""
    tpp5 = ""
    tpp6 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "tpp1" : tpp1, "tpp2" : tpp2, "tpp3" : tpp3, "tpp4" : tpp4, "tpp5" : tpp5, "tpp6": tpp6})

def applyTransLog(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = logarithmic_transformation(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "block"
    t2 = "none"
    t3 = "none"
    t4 = "none"
    t5 = "none"
    tpp1 = ""
    tpp2 = "checked"
    tpp3 = ""
    tpp4 = ""
    tpp5 = ""
    tpp6 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "tpp1" : tpp1, "tpp2" : tpp2, "tpp3" : tpp3, "tpp4" : tpp4, "tpp5" : tpp5, "tpp6": tpp6})


def applyInvImage(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = image_inverse(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "block"
    t2 = "none"
    t3 = "none"
    t4 = "none"
    t5 = "none"
    tpp1 = ""
    tpp2 = ""
    tpp3 = "checked"
    tpp4 = ""
    tpp5 = ""
    tpp6 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "tpp1" : tpp1, "tpp2" : tpp2, "tpp3" : tpp3, "tpp4" : tpp4, "tpp5" : tpp5, "tpp6": tpp6})


def applyHistoEgal(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = histogram_equalization(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "block"
    t2 = "none"
    t3 = "none"
    t4 = "none"
    t5 = "none"
    tpp1 = ""
    tpp2 = ""
    tpp3 = ""
    tpp4 = "checked"
    tpp5 = ""
    tpp6 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "tpp1" : tpp1, "tpp2" : tpp2, "tpp3" : tpp3, "tpp4" : tpp4, "tpp5" : tpp5, "tpp6": tpp6})


def applySeuil(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = global_thresholding(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "block"
    t2 = "none"
    t3 = "none"
    t4 = "none"
    t5 = "none"
    tpp1 = ""
    tpp2 = ""
    tpp3 = ""
    tpp4 = ""
    tpp5 = "checked"
    tpp6 = ""
    
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "tpp1" : tpp1, "tpp2" : tpp2, "tpp3" : tpp3, "tpp4" : tpp4, "tpp5" : tpp5, "tpp6": tpp6})


def applyDynRec(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = dynamic_range_adjustment(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "block"
    t2 = "none"
    t3 = "none"
    t4 = "none"
    t5 = "none"
    tpp1 = ""
    tpp2 = ""
    tpp3 = ""
    tpp4 = ""
    tpp5 = ""
    tpp6 = "checked"
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "tpp1" : tpp1, "tpp2" : tpp2, "tpp3" : tpp3, "tpp4" : tpp4, "tpp5" : tpp5, "tpp6": tpp6})



#filtrage spatial 
def applyTMoyeneur(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = filtre_Moy(image,N=3)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "block"
    t3 = "none"
    t4 = "none"
    t5 = "none"
    fs1 = "checked"
    fs2 = ""
    fs3 = ""
    fs4 = ""
    fs5 = ""
    fs6 = ""
    fs7 = ""
    fs8 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "fs1" : fs1, "fs2" : fs2, "fs3" : fs3, "fs4" : fs4, "fs5" : fs5, "fs6": fs6, "fs7": fs7, "fs8": fs8})

def applyBinomial(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = filtre_Binomial(image,N=3)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "block"
    t3 = "none"
    t4 = "none"
    t5 = "none"
    fs1 = ""
    fs2 = "checked"
    fs3 = ""
    fs4 = ""
    fs5 = ""
    fs6 = ""
    fs7 = ""
    fs8 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "fs1" : fs1, "fs2" : fs2, "fs3" : fs3, "fs4" : fs4, "fs5" : fs5, "fs6": fs6, "fs7": fs7, "fs8": fs8})

def applymedian(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = filtre_Median(image,N=3)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "block"
    t3 = "none"
    t4 = "none"
    t5 = "none"
    fs1 = ""
    fs2 = ""
    fs3 = "checked"
    fs4 = ""
    fs5 = ""
    fs6 = ""
    fs7 = ""
    fs8 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "fs1" : fs1, "fs2" : fs2, "fs3" : fs3, "fs4" : fs4, "fs5" : fs5, "fs6": fs6, "fs7": fs7, "fs8": fs8})

def applyMinMax(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = filtre_MinMax(image,N=3)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "block"
    t3 = "none"
    t4 = "none"
    t5 = "none"
    fs1 = ""
    fs2 = ""
    fs3 = ""
    fs4 = "checked"
    fs5 = ""
    fs6 = ""
    fs7 = ""
    fs8 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "fs1" : fs1, "fs2" : fs2, "fs3" : fs3, "fs4" : fs4, "fs5" : fs5, "fs6": fs6, "fs7": fs7, "fs8": fs8})

def applyNNS(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = nearest_neighbor_filter(image, window_size=3)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "block"
    t3 = "none"
    t4 = "none"
    t5 = "none"
    fs1 = ""
    fs2 = ""
    fs3 = ""
    fs4 = ""
    fs5 = "checked"
    fs6 = ""
    fs7 = ""
    fs8 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "fs1" : fs1, "fs2" : fs2, "fs3" : fs3, "fs4" : fs4, "fs5" : fs5, "fs6": fs6, "fs7": fs7, "fs8": fs8})

def applyNagao(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = nagao_filter(image, window_size=3)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "block"
    t3 = "none"
    t4 = "none"
    t5 = "none"
    fs1 = ""
    fs2 = ""
    fs3 = ""
    fs4 = ""
    fs5 = ""
    fs6 = "checked"
    fs7 = ""
    fs8 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "fs1" : fs1, "fs2" : fs2, "fs3" : fs3, "fs4" : fs4, "fs5" : fs5, "fs6": fs6, "fs7": fs7, "fs8": fs8})

def applyWiener(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = wiener_filter(image, kernel_size=5, noise_var=0.1)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "block"
    t3 = "none"
    t4 = "none"
    t5 = "none"
    fs1 = ""
    fs2 = ""
    fs3 = ""
    fs4 = ""
    fs5 = ""
    fs6 = ""
    fs7 = "checked"
    fs8 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "fs1" : fs1, "fs2" : fs2, "fs3" : fs3, "fs4" : fs4, "fs5" : fs5, "fs6": fs6, "fs7": fs7, "fs8": fs8})

def applyPhaut(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = filtre_PHaut(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "block"
    t3 = "none"
    t4 = "none"
    t5 = "none"
    fs1 = ""
    fs2 = ""
    fs3 = ""
    fs4 = ""
    fs5 = ""
    fs6 = ""
    fs7 = ""
    fs8 = "checked"
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "fs1" : fs1, "fs2" : fs2, "fs3" : fs3, "fs4" : fs4, "fs5" : fs5, "fs6": fs6, "fs7": fs7, "fs8": fs8})

#filtrage fruequentiel


def apply_Pb_ideal01(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = passe_bas_ideal_0_1(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "none"
    t3 = "block"
    t4 = "none"
    t5 = "none"
    ff1 = "checked"
    ff2 = ""
    ff3 = ""
    ff4 = ""
    ff5 = ""
    ff6 = ""
    ff7 = ""
    ff8 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "ff1" : ff1, "ff2" : ff2, "ff3" : ff3, "ff4" : ff4, "ff5" : ff5, "ff6": ff6, "ff7": ff7, "ff8": ff8})

def apply_Pb_ideal1_2(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = passe_bas_ideal_1demi_1(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "none"
    t3 = "block"
    t4 = "none"
    t5 = "none"
    ff1 = ""
    ff2 = "checked"
    ff3 = ""
    ff4 = ""
    ff5 = ""
    ff6 = ""
    ff7 = ""
    ff8 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "ff1" : ff1, "ff2" : ff2, "ff3" : ff3, "ff4" : ff4, "ff5" : ff5, "ff6": ff6, "ff7": ff7, "ff8": ff8})


def apply_Pb_Butterworth(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = passe_bas_butterworth(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "none"
    t3 = "block"
    t4 = "none"
    t5 = "none"
    ff1 = ""
    ff2 = ""
    ff3 = "checked"
    ff4 = ""
    ff5 = ""
    ff6 = ""
    ff7 = ""
    ff8 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "ff1" : ff1, "ff2" : ff2, "ff3" : ff3, "ff4" : ff4, "ff5" : ff5, "ff6": ff6, "ff7": ff7, "ff8": ff8})

def apply_Pb_Gaussien(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = passe_bas_gaussien(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "none"
    t3 = "block"
    t4 = "none"
    t5 = "none"
    ff1 = ""
    ff2 = ""
    ff3 = ""
    ff4 = "checked"
    ff5 = ""
    ff6 = ""
    ff7 = ""
    ff8 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "ff1" : ff1, "ff2" : ff2, "ff3" : ff3, "ff4" : ff4, "ff5" : ff5, "ff6": ff6, "ff7": ff7, "ff8": ff8})

def apply_Ph_ideal01(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = passe_Haut_ideal_0_1(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "none"
    t3 = "block"
    t4 = "none"
    t5 = "none"
    ff1 = ""
    ff2 = ""
    ff3 = ""
    ff4 = ""
    ff5 = "checked"
    ff6 = ""
    ff7 = ""
    ff8 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "ff1" : ff1, "ff2" : ff2, "ff3" : ff3, "ff4" : ff4, "ff5" : ff5, "ff6": ff6, "ff7": ff7, "ff8": ff8})

def apply_h_ideal1_2(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = passe_Haut_ideal_1demi_1(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "none"
    t3 = "block"
    t4 = "none"
    t5 = "none"
    ff1 = ""
    ff2 = ""
    ff3 = ""
    ff4 = ""
    ff5 = ""
    ff6 = "checked"
    ff7 = ""
    ff8 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "ff1" : ff1, "ff2" : ff2, "ff3" : ff3, "ff4" : ff4, "ff5" : ff5, "ff6": ff6, "ff7": ff7, "ff8": ff8})

def apply_Ph_Butterworth(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = passe_Haut_butterworth(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "none"
    t3 = "block"
    t4 = "none"
    t5 = "none"
    ff1 = ""
    ff2 = ""
    ff3 = ""
    ff4 = ""
    ff5 = ""
    ff6 = ""
    ff7 = "checked"
    ff8 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "ff1" : ff1, "ff2" : ff2, "ff3" : ff3, "ff4" : ff4, "ff5" : ff5, "ff6": ff6, "ff7": ff7, "ff8": ff8})

def apply_Ph_Gaussien(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = passe_Haut_gaussien(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "none"
    t3 = "block"
    t4 = "none"
    t5 = "none"
    ff1 = ""
    ff2 = ""
    ff3 = ""
    ff4 = ""
    ff5 = ""
    ff6 = ""
    ff7 = ""
    ff8 = "checked"
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "ff1" : ff1, "ff2" : ff2, "ff3" : ff3, "ff4" : ff4, "ff5" : ff5, "ff6": ff6, "ff7": ff7, "ff8": ff8})

#detectin de compteur 

def applyPrewit(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = prewitt_filter(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "none"
    t3 = "none"
    t4 = "block"
    t5 = "none"
    mdc1 = "checked"
    mdc2 = ""
    mdc3 = ""
    mdc4 = ""
    mdc5 = ""
    mdc6 = ""
    mdc7 = ""
    mdc8 = ""
    mdc9 = ""
    mdc10 = ""
    mdc11 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "mdc1" : mdc1, "mdc2" : mdc2, "mdc3" : mdc3, "mdc4" : mdc4, "mdc5" : mdc5, "mdc6": mdc6, "mdc7": mdc7, "mdc8": mdc8, "mdc9": mdc9, "mdc10": mdc10, "mdc11": mdc11})

def applyRoberts(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = roberts_filter(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "none"
    t3 = "none"
    t4 = "block"
    t5 = "none"
    mdc1 = ""
    mdc2 = "checked"
    mdc3 = ""
    mdc4 = ""
    mdc5 = ""
    mdc6 = ""
    mdc7 = ""
    mdc8 = ""
    mdc9 = ""
    mdc10 = ""
    mdc11 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "mdc1" : mdc1, "mdc2" : mdc2, "mdc3" : mdc3, "mdc4" : mdc4, "mdc5" : mdc5, "mdc6": mdc6, "mdc7": mdc7, "mdc8": mdc8, "mdc9": mdc9, "mdc10": mdc10, "mdc11": mdc11})

def applySobel(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = sobel_filter(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "none"
    t3 = "none"
    t4 = "block"
    t5 = "none"
    mdc1 = ""
    mdc2 = ""
    mdc3 = "checked"
    mdc4 = ""
    mdc5 = ""
    mdc6 = ""
    mdc7 = ""
    mdc8 = ""
    mdc9 = ""
    mdc10 = ""
    mdc11 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "mdc1" : mdc1, "mdc2" : mdc2, "mdc3" : mdc3, "mdc4" : mdc4, "mdc5" : mdc5, "mdc6": mdc6, "mdc7": mdc7, "mdc8": mdc8, "mdc9": mdc9, "mdc10": mdc10, "mdc11": mdc11})

def applysobel4Direct(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = sobel4Direct_filter(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "none"
    t3 = "none"
    t4 = "block"
    t5 = "none"
    mdc1 = ""
    mdc2 = ""
    mdc3 = ""
    mdc4 = "checked"
    mdc5 = ""
    mdc6 = ""
    mdc7 = ""
    mdc8 = ""
    mdc9 = ""
    mdc10 = ""
    mdc11 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "mdc1" : mdc1, "mdc2" : mdc2, "mdc3" : mdc3, "mdc4" : mdc4, "mdc5" : mdc5, "mdc6": mdc6, "mdc7": mdc7, "mdc8": mdc8, "mdc9": mdc9, "mdc10": mdc10, "mdc11": mdc11})

def applygradient(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = gradient_filter(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "none"
    t3 = "none"
    t4 = "block"
    t5 = "none"
    mdc1 = ""
    mdc2 = ""
    mdc3 = ""
    mdc4 = ""
    mdc5 = "checked"
    mdc6 = ""
    mdc7 = ""
    mdc8 = ""
    mdc9 = ""
    mdc10 = ""
    mdc11 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "mdc1" : mdc1, "mdc2" : mdc2, "mdc3" : mdc3, "mdc4" : mdc4, "mdc5" : mdc5, "mdc6": mdc6, "mdc7": mdc7, "mdc8": mdc8, "mdc9": mdc9, "mdc10": mdc10, "mdc11": mdc11})


def applygaussian(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = gaussian_filter(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "none"
    t3 = "none"
    t4 = "block"
    t5 = "none"
    mdc1 = ""
    mdc2 = ""
    mdc3 = ""
    mdc4 = ""
    mdc5 = ""
    mdc6 = "checked"
    mdc7 = ""
    mdc8 = ""
    mdc9 = ""
    mdc10 = ""
    mdc11 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "mdc1" : mdc1, "mdc2" : mdc2, "mdc3" : mdc3, "mdc4" : mdc4, "mdc5" : mdc5, "mdc6": mdc6, "mdc7": mdc7, "mdc8": mdc8, "mdc9": mdc9, "mdc10": mdc10, "mdc11": mdc11})

def applylaplacian_variance(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = laplacian_variance_filter(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "none"
    t3 = "none"
    t4 = "block"
    t5 = "none"
    mdc1 = ""
    mdc2 = ""
    mdc3 = ""
    mdc4 = ""
    mdc5 = ""
    mdc6 = ""
    mdc7 = "checked"
    mdc8 = ""
    mdc9 = ""
    mdc10 = ""
    mdc11 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "mdc1" : mdc1, "mdc2" : mdc2, "mdc3" : mdc3, "mdc4" : mdc4, "mdc5" : mdc5, "mdc6": mdc6, "mdc7": mdc7, "mdc8": mdc8, "mdc9": mdc9, "mdc10": mdc10, "mdc11": mdc11})

def applylaplacian_kirsch(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = kirsch_filter(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "none"
    t3 = "none"
    t4 = "block"
    t5 = "none"
    mdc1 = ""
    mdc2 = ""
    mdc3 = ""
    mdc4 = ""
    mdc5 = ""
    mdc6 = ""
    mdc7 = ""
    mdc8 = "checked"
    mdc9 = ""
    mdc10 = ""
    mdc11 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "mdc1" : mdc1, "mdc2" : mdc2, "mdc3" : mdc3, "mdc4" : mdc4, "mdc5" : mdc5, "mdc6": mdc6, "mdc7": mdc7, "mdc8": mdc8, "mdc9": mdc9, "mdc10": mdc10, "mdc11": mdc11})

def applylaplacian_robinson(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = robinson_filter(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "none"
    t3 = "none"
    t4 = "block"
    t5 = "none"
    mdc1 = ""
    mdc2 = ""
    mdc3 = ""
    mdc4 = ""
    mdc5 = ""
    mdc6 = ""
    mdc7 = ""
    mdc8 = ""
    mdc9 = "checked"
    mdc10 = ""
    mdc11 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "mdc1" : mdc1, "mdc2" : mdc2, "mdc3" : mdc3, "mdc4" : mdc4, "mdc5" : mdc5, "mdc6": mdc6, "mdc7": mdc7, "mdc8": mdc8, "mdc9": mdc9, "mdc10": mdc10, "mdc11": mdc11})

def applylocal_dispersion(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = local_dispersion_filter(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "none"
    t3 = "none"
    t4 = "block"
    t5 = "none"
    mdc1 = ""
    mdc2 = ""
    mdc3 = ""
    mdc4 = ""
    mdc5 = ""
    mdc6 = ""
    mdc7 = ""
    mdc8 = ""
    mdc9 = ""
    mdc10 = "checked"
    mdc11 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "mdc1" : mdc1, "mdc2" : mdc2, "mdc3" : mdc3, "mdc4" : mdc4, "mdc5" : mdc5, "mdc6": mdc6, "mdc7": mdc7, "mdc8": mdc8, "mdc9": mdc9, "mdc10": mdc10, "mdc11": mdc11})

def applyVariance(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = variance_filter(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "none"
    t3 = "none"
    t4 = "block"
    t5 = "none"
    mdc1 = ""
    mdc2 = ""
    mdc3 = ""
    mdc4 = ""
    mdc5 = ""
    mdc6 = ""
    mdc7 = ""
    mdc8 = ""
    mdc9 = ""
    mdc10 = ""
    mdc11 = "checked"
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "mdc1" : mdc1, "mdc2" : mdc2, "mdc3" : mdc3, "mdc4" : mdc4, "mdc5" : mdc5, "mdc6": mdc6, "mdc7": mdc7, "mdc8": mdc8, "mdc9": mdc9, "mdc10": mdc10, "mdc11": mdc11})




#Morphology
def applyerosion(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = erosion(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "none"
    t3 = "none"
    t4 = "none"
    t5 = "block"
    mrp1 = "checked"
    mrp2 = ""
    mrp3 = ""
    mrp4 = ""
    mrp5 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "mrp1" : mrp1, "mrp2" : mrp2, "mrp3" : mrp3, "mrp4" : mrp4, "mrp5" : mrp5})

def applydilation(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = dilation(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "none"
    t3 = "none"
    t4 = "none"
    t5 = "block"
    mrp1 = ""
    mrp2 = "checked"
    mrp3 = ""
    mrp4 = ""
    mrp5 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "mrp1" : mrp1, "mrp2" : mrp2, "mrp3" : mrp3, "mrp4" : mrp4, "mrp5" : mrp5})

def applyouverture(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = ouverture(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "none"
    t3 = "none"
    t4 = "none"
    t5 = "block"
    mrp1 = ""
    mrp2 = ""
    mrp3 = "checked"
    mrp4 = ""
    mrp5 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "mrp1" : mrp1, "mrp2" : mrp2, "mrp3" : mrp3, "mrp4" : mrp4, "mrp5" : mrp5})

def applytout_ou_rien(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = tout_ou_rien(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "none"
    t3 = "none"
    t4 = "none"
    t5 = "block"
    mrp1 = ""
    mrp2 = ""
    mrp3 = ""
    mrp4 = "checked"
    mrp5 = ""
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "mrp1" : mrp1, "mrp2" : mrp2, "mrp3" : mrp3, "mrp4" : mrp4, "mrp5" : mrp5})

def applysegmentation_seuillage(request): 
    if request.method == "POST":
        image, x_size1, y_size1 = load_image_data(os.path.join(settings.STATIC_ROOT, request.session['image_noise_path']))
        n = segmentation_seuillage(image)
        if(request.session['image_out_path'] == request.session['image_in_path']):
            request.session['image_out_path'] = f"images/{request.session['id']}_output.jpg"
        save_image_data(n, os.path.join(settings.STATIC_ROOT, request.session['image_out_path']))
    t1 = "none"
    t2 = "none"
    t3 = "none"
    t4 = "none"
    t5 = "block"
    mrp1 = ""
    mrp2 = ""
    mrp3 = ""
    mrp4 = ""
    mrp5 = "checked"
    return render(request, "index.html", {"image_noise_path": request.session['image_noise_path'], "image_out_path": request.session['image_out_path'], "t1" : t1, "t2" : t2, "t3" : t3, "t4" : t4, "t5" : t5, "mrp1" : mrp1, "mrp2" : mrp2, "mrp3" : mrp3, "mrp4" : mrp4, "mrp5" : mrp5})


