from __future__ import unicode_literals
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import segmentation
from os import listdir
from os.path import isfile, join


 #["ka","kha","ga","gha","kna","cha","chha","ja","jha","yna","t`a","t`ha","d`a","d`ha","adna","ta","tha","da","dha","na","pa","pha","ba","bha","ma","yaw","ra","la","waw","sha","shat","sa","ha","aksha","tra","gya","0","1","2","3","4","5","6","7","8","9"]
labels =['yna', 't`aa', 't`haa', 'd`aa', 'd`haa', 'a`dna', 'ta', 'tha', 'da', 'dha', 'ka', 'na', 'pa', 'pha', 'ba', 'bha', 'ma', 'yaw', 'ra', 'la', 'waw', 'kha', 'sha', 'shat', 'sa', 'ha', 'aksha', 'tra', 'gya', 'ga', 'gha', 'kna', 'cha', 'chha', 'ja', 'jha', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

#labels = [u'\u091E',u'\u091F',u'\u0920',u'\u0921',u'\u0922',u'\u0923',u'\u0924',u'\u0925',u'\u0926',u'\u0927',u'\u0915',u'\u0928',u'\u092A',u'\u092B',u'\u092c',u'\u092d',u'\u092e',u'\u092f',u'\u0930',u'\u0932',u'\u0935',u'\u0916',u'\u0936',u'\u0937',u'\u0938',u'\u0939','ksha','tra','gya',u'\u0917',u'\u0918',u'\u0919',u'\u091a',u'\u091b',u'\u091c',u'\u091d',u'\u0966',u'\u0967',u'\u0968',u'\u0969',u'\u096a',u'\u096b',u'\u096c',u'\u096d',u'\u096e',u'\u096f']
#
#labels = ["ka","kha","ga","gha","kna","cha","chha","ja","jha","yna","t`a","t`ha","d`a","d`ha","adna","ta","tha","da","dha","na","pa","pha","ba","bha","ma","yaw","ra","la","waw","sha","shat","sa","ha","aksha","tra","gya","0","1","2","3","4","5","6","7","8","9"]

import numpy as np
from keras.preprocessing import image

mypath ="C:/Users/ELCOT-Lenovo/Downloads/Hindi-OCR-master/Hindi-OCR-master/result"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f))]
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    images[n] = cv2.imread(join(mypath, onlyfiles[n]))

print("word list",segmentation.wordImgList)
#test_image = cv2.imread(segmentation.wordImgList)




#image = cv2.imread("filename") 
#image = cv2.fastNlMeansDenoisingColored(test_image,None,10,10,7,21)
#gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#res,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV) #threshold 
#kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)) 

#dilated = cv2.dilate(thresh,kernel,iterations = 5) 

#contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 

#coord = []
#for contour in contours:  
#      [x,y,w,h] = cv2.boundingRect(contour)   
#      if h>300 and w>300:   
#          continue   
#      if h<40 or w<40:   
#          continue  
#      coord.append((x,y,w,h)) 

#coord.sort(key=lambda tup:tup[0]) # if the image has only one sentence sort in one axis

#count = 0
#for cor in coord:
#        [x,y,w,h] = cor
#        t = image[y:y+h,x:x+w,:]
#        cv2.imwrite(str(count)+".png",t)
#print("number of char in image:", count)



  
image = cv2.resize(images[0], (32,32))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = np.expand_dims(image, axis=0)
image = np.expand_dims(image, axis=3)
print("[INFO] loading network...")
import tensorflow as tf
model = tf.keras.models.load_model("C:/Users/ELCOT-Lenovo/Downloads/Hindi-OCR-master/Hindi-OCR-master/HindiModel2.h5")
lists = model.predict(image)[0]
#result[i]=labels[np.argmax(lists)]
print("The letter is ",labels[np.argmax(lists)])
	
