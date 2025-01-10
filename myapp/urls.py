from django.urls import path
from . import views

urlpatterns = [
    path('', views.home),
    path('upload/', views.upload),
    path('gaussien-noise/', views.applyNoiseGaussien),
    path('salt-pepper/', views.applyNoiseSaltPepper),
    path('periodic-noise/', views.applyNoisePeriodic),
    path('quantization-noise/', views.applyNoiseQuantisation),
    path('poisson-noise/', views.applyNoisePoisson),
    path('speckle-noise/', views.applyNoiseSpeckle),
    path('impulse-noise/', views.applyNoiseImpulse),

#revenir à l'image originale
    path('image_original_grise/', views.applyimage_original_grise),
    path('image_original/', views.applyimage_original),

#Traitmen par point
    path('trans-exp/', views.applyTransExp),
    path('trans-log/', views.applyTransLog),
    path('inv-image/', views.applyInvImage),
    path('histo-egal/', views.applyHistoEgal),
    path('seuillage/', views.applySeuil),
    path('dyn-recadrage/', views.applyDynRec),

#filtrage spatial 
    path('Moyenneur/', views.applyTMoyeneur),
    path('Binomial/', views.applyBinomial),
    path('median/', views.applymedian),
    path('MinMax/', views.applyMinMax),
    path('Nearest_Neighbor/', views.applyNNS),
    path('filtre_de_Nagao/', views.applyNagao),
    path('Wiener/', views.applyWiener),
    path('passe-haut/', views.applyPhaut),

#filtrage fruequentiel
    path('Passe-bas-ideal-0|1/', views.apply_Pb_ideal01),
    path('Passe-bas-ideal-1/2|1/', views.apply_Pb_ideal1_2),
    path('Passe-bas-Butterworth/', views.apply_Pb_Butterworth),
    path('Passe-bas-Gaussien/', views.apply_Pb_Gaussien),
    path('Passe-haut ideal-0|1/', views.apply_Ph_ideal01),
    path('Passe-haut-ideal-1/2|1/', views.apply_h_ideal1_2),
    path('Passe-haut-Butterworth/', views.apply_Ph_Butterworth),
    path('Passe-haut-Gaussien/', views.apply_Ph_Gaussien),


#detectin de compteur 
    path('Prewit/', views.applyPrewit),
    path('Roberts/', views.applyRoberts),
    path('Sobel/', views.applySobel),
    path('sobel4Direct/', views.applysobel4Direct),
    path('gradient/', views.applygradient),
    path('gaussian/', views.applygaussian),
    path('laplacian-variance/', views.applylaplacian_variance),
    path('laplacian-kirsch/', views.applylaplacian_kirsch),
    path('laplacian-robinson/', views.applylaplacian_robinson),
    path('local-dispersion/', views.applylocal_dispersion),
    path('Sobel-gradient/', views.applyVariance),



#Morphology
    path('erosion/', views.applyerosion),
    path('dilation/', views.applydilation),
    path('ouverture/', views.applyouverture),
    path('tout_ou_rien/', views.applytout_ou_rien),
    path('segmentation-seuillage/', views.applysegmentation_seuillage),
   
]