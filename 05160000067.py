#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from skimage import filters, exposure, transform, morphology
from skimage import data
import cv2 as cv
import numpy as np
import tkinter as tk 
from tkinter import filedialog 
import random 
import os
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage.segmentation import active_contour


root = tk.Tk() 
root.withdraw()
video_control = False
class MyApp():
    def __init__(self): # StartPoint   
        image = MyApp.getImage()
        if video_control != True:
            MyApp.editPhoto(image)

    def savePhoto(self):
        path = input("""Resmi kaydetmek istediğiniz dosyanın uzantısını giriniz. (Örn: "C:"\"Users"\"aygen")\n""")
        cv.imwrite(os.path.join(path , 'image.jpg'), image)
        print("Fotoğraf başarıyla kaydedildi!") 
        cv.waitKey(0)
        exit()

    def askContinue(image): 
        ContinueOpt = int(input("""Nasıl devam etmek istersiniz?:\n
                      [1] Aynı fotoğraf üzerinden işlem yapmaya devam et.
                      [2] Ana menüye dön.
                      [3] Fotoğrafı kaydet.
                      [4] Programdan çık. \n"""))
        if ContinueOpt == 1:
            MyApp.editPhoto(image)
        if ContinueOpt == 2:
            MyApp()
        if ContinueOpt == 3:
            MyApp.savePhoto(image)
        if ContinueOpt == 4:
            exit()  

    def getImage():
        getImageOpt = int(input("""Yapmak istediğiniz işlemi seçiniz. :\n
                    [1] Yeni bir fotoğraf yükle.
                    [2] Kütüphanede mevcut olan fotoğraflar ile devam et.
                    [3] Video yükle(Yalnızca kenar belirleme işlemi yapar).\n"""))

        if getImageOpt == 1:
            file_path = filedialog.askopenfilename() 
            image = cv.imread(file_path)
            print("Fotoğraf başarıyla yüklendi!\n") 
            return image
        if getImageOpt == 2:
            image = input("""Kütüphaneden yüklemek istediğiniz fotoğrafın adını giriniz.("coins, cameramen, cell, horse vb.")\n """)
            image = getattr(data, image)()
            print("Fotoğraf başarıyla yüklendi!\n") 
            return image        
        if getImageOpt == 3:
            global video_control
            video_control = True
            file_path = filedialog.askopenfilename() 
            capture = cv.VideoCapture(file_path)
            while True:
                _, frame = capture.read()
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                edges = cv.Canny(gray, 50, 150) # Find edges for the image
                cv.imshow('frame', frame)
                cv.imshow('edges', edges)
                k = cv.waitKey(25)
                if k == 27:
                    break        
            cv.destroyAllWindows()
            MyApp()


    def editPhoto(image):

        process = int(input("""Fotoğraf üzerinde yapmak istediğiniz işlemini seçiniz:\n
                    [1] Filtreleme
                    [2] Histogram görüntüleme ve işleme
                    [3] Uzaysal dönüşüm işlemleri ("Resizing, rotating, cropping, vb.")
                    [4] Yoğunluk dönüşümü işlemleri 
                    [5] Morfolojik işlemler
                    [6] Active countor örneği
                    [7] Instagram filtresi\n"""))

        if process == 1: #...filtreleme
            if len(image.shape) == 3:
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  
            filter = int(input("""Kullanmak istediğiniz filtreleme işlemini seçiniz:\n
               [1]Gaussian
               [2]Sobel
               [3]Diag
               [4]Gabor
               [5]Meijering
               [6]Sato
               [7]Hessian
               [8]SobelV
               [9]SobelH
               [10]Scharr\n"""))    
            if filter == 1:  
                sigma = float(input("(Sigma) parametreyi giriniz: "))
                gaussian_img  = filters.gaussian(image,sigma)
                cv.imshow('image_orijinal', image)
                cv.imshow('gaussian_img ', gaussian_img)
                cv.waitKey(0)
                cv.destroyAllWindows()
                MyApp.askContinue(image)
            if filter == 2:
                sobel_img = filters.sobel(image)
                cv.imshow('image_orijinal', image)
                cv.imshow('sobel_img ', sobel_img)
                cv.waitKey(0)
                cv.destroyAllWindows()
                MyApp.askContinue(image)
            if filter == 3:
                diag_img      = filters.roberts_neg_diag(image)
                cv.imshow('image_orijinal', image)
                cv.imshow('diag_img', diag_img)
                cv.waitKey(0)
                cv.destroyAllWindows()
                MyApp.askContinue(image)
            if filter == 4:
                gabor_img,_   = filters.gabor(image,frequency=0.6)
                cv.imshow('image_orijinal', image)
                cv.imshow('gabor_img', gabor_img)
                cv.waitKey(0)
                cv.destroyAllWindows()
                MyApp.askContinue(image)
            if filter == 5:
                meijering_img = filters.meijering(image,alpha=2)
                cv.imshow('image_orijinal', image)
                cv.imshow('meijering_img', meijering_img)
                cv.waitKey(0)
                cv.destroyAllWindows()
                MyApp.askContinue(image)
            if filter == 6:
                sato_img      = filters.sato(image)
                cv.imshow('image_orijinal', image)
                cv.imshow('sato_img ', sato_img)
                cv.waitKey(0)
                cv.destroyAllWindows()
                MyApp.askContinue(image)
            if filter == 7:
                hessian_img   = filters.hessian(image)
                cv.imshow('image_orijinal', image)
                cv.imshow('hessian_img ', hessian_img)
                cv.waitKey(0)
                cv.destroyAllWindows()
                MyApp.askContinue(image)
            if filter == 8:
                sobelv_img    = filters.sobel_v(image)
                cv.imshow('image_orijinal', image)
                cv.imshow('sobelv_img', sobelv_img)
                cv.waitKey(0)
                cv.destroyAllWindows()
                MyApp.askContinue(image)
            if filter == 9:
                sobelh_img    = filters.sobel_h(image)
                cv.imshow('image_orijinal', image)
                cv.imshow('sobelh_img', sobelh_img)
                cv.waitKey(0)
                cv.destroyAllWindows()
                MyApp.askContinue(image)
            if filter == 10:
                scharr_img    = filters.scharr(image)
                cv.imshow('image_orijinal', image)
                cv.imshow('scharr_img', scharr_img)
                cv.waitKey(0)
                cv.destroyAllWindows()
                MyApp.askContinue(image)

        if process == 2: #...histogram

            hist, hist_centers = exposure.histogram(image)
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))

            axes[0].imshow(image, cmap=plt.cm.gray)
            axes[0].axis('off')
            axes[1].plot(hist_centers, hist, lw=2)
            axes[1].set_title('histogram of gray values')

            img_eq = exposure.equalize_hist(image)
            hist2, hist_centers2 = exposure.histogram(img_eq)
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))

            axes[0].imshow(img_eq, cmap=plt.cm.gray)
            axes[0].axis('off')
            axes[1].plot(hist_centers2, hist2, lw=2)
            axes[1].set_title('histogram of gray values')
            cv.waitKey(0)
            MyApp.askContinue(image)


        if process == 3: #...uzaysal dönüşüm

            part5 = int(input("""Hangi uzaysal dönüşüm işlemini yapmak istiyorsunuz?:\n
                          [1]Rescaled
                          [2]Swirled
                          [3]Cropping
                          [4]Resizing
                          [5]Rotation\n"""))

            if part5 == 1: #...Rescaled
                rescaled = float(input("Rescaled parametresini giriniz (recommended = 0.25): "))
                image_rescaled = transform.rescale(image, rescaled, anti_aliasing=False)
                cv.imshow('image_orijinal', image)
                cv.imshow('image_rescaled',image_rescaled)
                print("Rescaling işlemi yapıldı!")
                cv.waitKey(0)
                cv.destroyAllWindows()
                MyApp.askContinue(image_rescaled)


            if part5 == 2: #...Swirled
                rotation_ = int(input("Rotation parametresini giriniz: "))
                strength_ = int(input("Strength parametresini giriniz: "))
                radius_ = int(input("Radius parametresini giriniz: "))
                swirled = transform.swirl(image, rotation=rotation_, strength=strength_, radius=radius_)
                cv.imshow('image_orijinal', image)
                cv.imshow('swirled',swirled)
                print("Swirling işlemi başarıyla yapıldı!")
                cv.waitKey(0)
                cv.destroyAllWindows()
                MyApp.askContinue(swirled)

            if part5 == 3: #...Cropping
                print("Fotoğraf üzerinde kırpmak istediğiniz alanı seçiniz.\n")
                roi = cv.selectROI(image)
                print("Kırpmak için ESC'ye basınız.")
                im_cropped = image[int(roi[1]):int(roi[1]+roi[3]),int(roi[0]):int(roi[0]+roi[2])]
                print("Cropping işlemi başarıyla yapıldı!")
                cv.imshow('image_orijinal', image)
                cv.imshow("Cropped Image", im_cropped)
                cv.waitKey(0)
                cv.destroyAllWindows()
                MyApp.askContinue(im_cropped)


            if part5 == 4: #....Resizing
                row = int(input("Yükseliği giriniz: "))
                col = int(input("Genişliği giriniz: "))
                image_resized = transform.resize(image, (row, col))
                cv.imshow('image_orijinal', image)
                cv.imshow('image_resized',image_resized)
                print("Resizing işlemi başarıyla yapıldı!")
                cv.waitKey(0)
                cv.destroyAllWindows()
                MyApp.askContinue(image_resized)

            if part5 == 5: #...Rotation
                tform = transform.AffineTransform(scale=(1.5, 1.5), rotation=0.5,
                                       translation=(150, -200))
                img_warp = transform.warp(image, tform)
                cv.imshow('image_orijinal', image)
                cv.imshow('img_warp',img_warp)
                print("Rotating işlemi başarıyla yapıldı!")
                cv.waitKey(0)
                cv.destroyAllWindows()
                MyApp.askContinue(img_warp)

        if process == 4: #...yoğunluk dönüşümü
            cv.namedWindow("Yogunluk Kontrol")
            def nothing(x):
                pass
            cv.createTrackbar("range1","Yogunluk Kontrol",0,255,nothing)
            cv.createTrackbar("range2","Yogunluk Kontrol",0,255,nothing)

            while True:
                range1_ = cv.getTrackbarPos("range1","Yogunluk Kontrol")
                range2_ = cv.getTrackbarPos("range2","Yogunluk Kontrol")
                img = exposure.rescale_intensity(image,in_range=(range1_, range2_))
                cv.imshow('img',img)
                cv.imshow('image',image)
                k = cv.waitKey(25)
                if k == 27:
                    break
            cv.destroyAllWindows()
            MyApp.askContinue(image)

        if process == 5: #...morfolojik işlemler
            part = int(input("""Yapmak istediğiniz morfolojik işlemi seçiniz:\n
                              [1] Erosion
                              [2] Dilation
                              [3] Opening
                              [4] Closing"""))
            if part == 1: #..Erosion
                image2 = morphology.erosion(image)
                cv.imshow('image_orijinal', image)
                cv.imshow('image2',image2)
                print("Çıkmak için herhangi bir tuşa basınız!")
                cv.waitKey(0)
                cv.destroyAllWindows()
                MyApp.askContinue(image)
                
            if part == 2: #..dilation
                image3 = morphology.dilation(image)
                cv.imshow('image_orijinal', image)
                cv.imshow('image3',image3)
                print("Çıkmak için herhangi bir tuşa basınız!")
                cv.waitKey(0)
                cv.destroyAllWindows()
                MyApp.askContinue(image)
                
            if part == 3: #...Opening
                image4 = morphology.opening(image)
                cv.imshow('image_orijinal', image)
                cv.imshow('image4',image4)
                print("Çıkmak için herhangi bir tuşa basınız!")
                cv.waitKey(0)
                cv.destroyAllWindows()
                MyApp.askContinue(image)
                
            if part == 4: #...closing
                image5 = morphology.closing(image)
                cv.imshow('image_orijinal', image)
                cv.imshow('image5',image5)
                print("Çıkmak için herhangi bir tuşa basınız!")
                cv.waitKey(0)
                cv.destroyAllWindows()
                MyApp.askContinue(image)
                
        if process == 6: #...active countor
            r = np.linspace(136, 50, 100)
            c = np.linspace(5, 424, 100)
            init = np.array([r, c]).T

            snake = active_contour(gaussian(image, 1), init, boundary_condition='fixed',
                                   alpha=0.3, beta=1.0, w_line=-5, w_edge=0, gamma=0.1)

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.imshow(image, cmap=plt.cm.gray)
            ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
            ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
            ax.set_xticks([]), ax.set_yticks([])
            ax.axis([0, image.shape[1], image.shape[0], 0])

            plt.show()
            MyApp()
            
        if process == 7: #...instagram
            image_ = image.copy()
            cv.namedWindow("instagram")
            def nothing(x):
                pass
            cv.createTrackbar("Grey","instagram",0,1,nothing)
            cv.createTrackbar("Scharr","instagram",0,1,nothing)
            cv.createTrackbar("range1 for morphology","instagram",0,255,nothing)
            cv.createTrackbar("range2 for morphology","instagram",0,255,nothing)

            while True:
                image = image_
                Grey     = cv.getTrackbarPos("Grey","instagram")
                Scharr   = cv.getTrackbarPos("Scharr","instagram")
                range1_  = cv.getTrackbarPos("range1 for morphology","instagram")
                range2_  = cv.getTrackbarPos("range2 for morphology","instagram")

                if Scharr == 1:
                    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                    image = filters.scharr(image)
                elif Grey == 1:
                    if len(image.shape) != 2:
                        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                if range1_ != 0 or range2_ != 0:
                    image = exposure.rescale_intensity(image,in_range=(range1_, range2_))

                cv.imshow('image',image)
                k = cv.waitKey(25)
                if k == 27: # ESC
                    break
            cv.destroyAllWindows()
            MyApp.askContinue(image)

if __name__ == "__main__":
    MyApp()


# In[ ]:




