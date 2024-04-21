import cv2
import numpy as np
import matplotlib.pyplot as plt
import card_identify


def ORB(frame):
    orb = cv2.ORB_create()
    keypoints = orb.detect(frame,None)

    kepypoints,__ = orb.compute(frame,keypoints)
    imgGray = cv2.drawKeypoints(frame, keypoints, frame,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return imgGray
    # plt.figure()
    # plt.imshow(imgGray)
    # plt.show()

def get_shapes(isolated_card_img):
    gray = cv2.cvtColor(isolated_card_img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(1,1),1000)
    flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

    # In the orignal thresh, the shapes were black and the outside was white, but we flip it so the only thing outlining in our contours are the shapes, not the border of the image itself.
    thresh = cv2.bitwise_not(thresh)

    # Contours are sets of corner coordinates
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea,reverse=True)[:3]

    return contours

    # Compare Contours using the match shapes



video = cv2.VideoCapture(0)
while True:
    frame = video.read()[1]
    # frame = cv2.imread("cards/testing.jpg")
    card_imgs = card_identify.get_card_imgs(frame, 2)
    # try:
    #     card_identify.show_cards(card_imgs)
    # except:
    #     print("bad_image")
  
    if len(card_imgs) > 0:
        # frame = cv2.imread("cards/2_oval_line.jpg")
        try:
            cv2.drawContours(card_imgs[0], get_shapes(card_imgs[0]), -1, (0,255,0), 5)
            cv2.imshow("success", card_imgs[0])
        except:
            print("bad")
            #  cv2.imshow("failed", card_imgs[0])

    if cv2.waitKey(60) & 0xFF == ord('q'):
            break
    

    
    

