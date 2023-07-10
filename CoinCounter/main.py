import cv2
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

sum = 0

myColorFinder = ColorFinder(True)
hsvVals = {'hmin': 0, 'smin': 0, 'vmin': 145, 'hmax': 63, 'smax': 91, 'vmax': 255}

def empty(a):
    pass


cv2.namedWindow("Settings")
cv2.resizeWindow("Settings", 640, 240)
cv2.createTrackbar("Threshold1", "Settings", 50, 255, empty)
cv2.createTrackbar("Threshold2", "Settings", 100, 255, empty)


def preProcessing(img):
    imgPre = cv2.GaussianBlur(img, (5, 5), 3)
    thresh1 = cv2.getTrackbarPos("Threshold1", "Settings")
    thresh2 = cv2.getTrackbarPos("Threshold2", "Settings")
    imgPre = cv2.Canny(imgPre, thresh1, thresh2)
    kernel = np.ones((3, 3), np.uint8)
    imgPre = cv2.dilate(imgPre, kernel, iterations=1)
    imgPre = cv2.morphologyEx(imgPre, cv2.MORPH_CLOSE, kernel)

    return imgPre


while True:
    success, img = cap.read()
    imgPre = preProcessing(img)
    imgContours, conFound = cvzone.findContours(img, imgPre, minArea=20)

    sum = 0
    if conFound:
        for count, contour in enumerate(conFound):
            peri = cv2.arcLength(contour['cnt'], True)
            approx = cv2.approxPolyDP(contour['cnt'], 0.02*peri, True)
            if len(approx)>5:
                area = contour['area']
                x,y,w,h=contour['bbox']
                imgCrop = img[y:y+h, x:x+w]
                imgColor, mask = myColorFinder.update(imgCrop, hsvVals)
                whitePixCount = cv2.countNonZero(mask)
                if area<2200:
                    sum += 1
                elif 2230 < area < 2370:
                    sum += 10
                elif 2390 < area < 2480:
                    sum += 2
                elif 2550 < area:
                    sum +=5


    #print(sum)
    imgStacked = cvzone.stackImages([img], 1, 1)
    cvzone.putTextRect(imgStacked, f'Rubs{sum}',(50,50))
    cv2.imshow("Image", imgStacked)


    cv2.waitKey(1)