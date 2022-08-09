import cv2
import cvlib as cv
import sys
import numpy as np
#img = cv2.imread('aa.webp')
img = cv2.imread('a.jpg')
face, confidence = cv.detect_face(img)
padding = 20
print(face)
print('-'*20)
print(len(face))
print('-'*20)
print(confidence)
for idx, f in enumerate(face):
        
    (startX,startY) = max(0, f[0]-padding), max(0, f[1]-padding)
    (endX,endY) = min(img.shape[1]-1, f[2]+padding), min(img.shape[0]-1, f[3]+padding)
    
    # draw rectangle over face
    cv2.rectangle(img, (startX,startY), (endX,endY), (0,255,0), 2)
    face_crop = np.copy(img[startY:endY, startX:endX])
    (label, confidence) = cv.detect_gender(face_crop)
    print(confidence)
    print(label)
    idx = np.argmax(confidence)
    label = label[idx]
    label = "{}: {:.2f}%".format(label, confidence[idx] * 100)
    Y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(img, label, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0), 2)
    
cv2.imshow('image window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
sys.exit()
def resizer(img,scale_percent):
#    scale_percent = 60 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

resized = resizer(img,10)

face, confidence = cv.detect_face(resized)
print(face)
print('-'*20)
print(len(face))
print('-'*20)
print(confidence)
cv2.imshow('image window', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()



