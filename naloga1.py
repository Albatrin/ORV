import cv2 as cv
import numpy as np
import time
def zmanjsaj_sliko(slika, sirina, visina):
    return cv.resize(slika, (sirina, visina))

def obdelaj_sliko_s_skatlami(slika, sirina_skatle, visina_skatle, barva_koze) -> list:

    spodnja_meja, zgornja_meja = barva_koze
    skatle = []
    h, w = slika.shape[:2]
    
    for y in range(0, h, visina_skatle):
        for x in range(0, w, sirina_skatle):
            x_end = min(x + sirina_skatle, w)
            y_end = min(y + visina_skatle, h)
            okno = slika[y:y_end, x:x_end]
            piksli_koze = prestej_piklse_z_barvo_koze(okno, (spodnja_meja, zgornja_meja))
            skatle.append((x, y, piksli_koze))
    
    return skatle


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
    velikost_skatle=15
    kamera = cv.VideoCapture(0)
    if not kamera.isOpened():
        print("Napaka pri odpiranju kamere")
        exit()

    ret, frame = kamera.read()
    if not ret:
        print("Napaka pri branju slike")
        kamera.release()
        exit()
    # obmocje
    h, w, _ = frame.shape
    levo_zgoraj = (w // 2 - 20, h // 2 - 20)
    desno_spodaj = (w // 2 + 20, h // 2 + 20)

    barva_koze = doloci_barvo_koze(frame, levo_zgoraj, desno_spodaj)

    while True:

        ret, frame = kamera.read()
        frame = cv.flip(frame, 1)
        if not ret:
            break

        frame = zmanjsaj_sliko(frame, 640, 480)
        skatle = obdelaj_sliko_s_skatlami(frame, velikost_skatle, velikost_skatle, barva_koze)

        prag = int(velikost_skatle * velikost_skatle * 0.10)

        for x, y, piksli_koze in skatle:
            if piksli_koze > prag:
                cv.rectangle(frame, (x, y), (x + velikost_skatle, y + velikost_skatle), (0, 255, 0), 1)

       
        cv.imshow('Detekcija obraza', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    kamera.release()
    cv.destroyAllWindows()