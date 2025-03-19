import cv2 as cv
import numpy as np

def zmanjsaj_sliko(slika, sirina, visina):
    return cv.resize(slika,(sirina,visina))

def obdelaj_sliko_s_skatlami(slika, sirina_skatle, visina_skatle, barva_koze) -> list:
    '''Sprehodi se skozi sliko v velikosti škatle (sirina_skatle x visina_skatle) in izračunaj število pikslov kože v vsaki škatli.
    Škatle se ne smejo prekrivati!
    Vrne seznam škatel, s številom pikslov kože.
    Primer: Če je v sliki 25 škatel, kjer je v vsaki vrstici 5 škatel, naj bo seznam oblike
      [[1,0,0,1,1],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[1,0,0,0,1]]. 
      V tem primeru je v prvi škatli 1 piksel kože, v drugi 0, v tretji 0, v četrti 1 in v peti 1.'''
    res=[]
    visina, sirina, _ = slika.shape #dimenzija celotne slike
    for y in range(0,visina,visina_skatle):
        vrstica=[]
        for x in range(0,sirina,sirina_skatle):
            x2 = min(x + sirina_skatle, sirina)  
            y2 = min(y + visina_skatle, visina)  
            roi = slika[y:y2, x:x2]
            piksli_koze = prestej_piklse_z_barvo_koze(roi, barva_koze)
            vrstica.append(piksli_koze)

            res.append(vrstica)            
   
            
    return res

def prestej_piklse_z_barvo_koze(slika, barva_koze) -> int:
    '''Prestej število pikslov z barvo kože v škatli.'''
    spodnja_meja, zgornja_meja = barva_koze  # Dobi meje kože
    maska = cv.inRange(slika, spodnja_meja, zgornja_meja) 
    return np.sum(maska == 255)  

def doloci_barvo_koze(slika,levo_zgoraj,desno_spodaj) -> tuple:
    '''Ta funkcija se kliče zgolj 1x na prvi sliki iz kamere. 
    Vrne barvo kože v območju ki ga definira oklepajoča škatla (levo_zgoraj, desno_spodaj).
      Način izračuna je prepuščen vaši domišljiji.'''
    x1,y1=levo_zgoraj
    x2,y2=desno_spodaj
    roi =slika[y1:y2,x1:x2]
    average=np.mean(roi, axis=(0,1))
    toleranca=20
    spodnja_meja = np.clip(average - toleranca, 0, 255).astype(np.uint8)
    zgornja_meja = np.clip(average + toleranca, 0, 255).astype(np.uint8)
    return(spodnja_meja,zgornja_meja)

if __name__ == '__main__':
    #Pripravi kamero
    kamera = cv.VideoCapture(0)
    slika=kamera.read()

    #Zajami prvo sliko iz kamere
    
    #Izračunamo barvo kože na prvi sliki
    
    #Zajemaj slike iz kamere in jih obdeluj     
    
    #Označi območja (škatle), kjer se nahaja obraz (kako je prepuščeno vaši domišljiji)
        #Vprašanje 1: Kako iz števila pikslov iz vsake škatle določiti celotno območje obraza (Floodfill)?
        #Vprašanje 2: Kako prešteti število ljudi?

        #Kako velikost prebirne škatle vpliva na hitrost algoritma in točnost detekcije? Poigrajte se s parametroma velikost_skatle
        #in ne pozabite, da ni nujno da je škatla kvadratna.
    pass