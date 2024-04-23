import cv2
import numpy as np
import matplotlib.pyplot as plt
import find_cards




def get_shape_contours(isolated_card_img):
    gray = cv2.cvtColor(isolated_card_img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(1,1),1000)
    flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_OTSU)

    # In the orignal thresh, the shapes were black and the outside was white, but we flip it so the only thing outlining in our contours are the shapes, not the border of the image itself.
    thresh = cv2.bitwise_not(thresh)

    # Contours are sets of corner coordinates
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea,reverse=True)[:3]
         
    return contours

def get_count(shape_contours):
     return len(shape_contours)

def get_shape_type(image_shape):
    shapes = ["diamond", "oval", "wave"]
    shape_contours = {}

    for shape in shapes:
        shape_img = cv2.imread("cards/" + shape + ".jpg")
        shape_contours[shape] = get_shape_contours(shape_img)[0]

    best_match = .2
    best_shape = "No Match"

    for shape in shape_contours:
        match_score = cv2.matchShapes(image_shape, shape_contours[shape], 1, 0.0)

        if match_score < best_match:
            best_match = match_score
            best_shape = shape

    return best_shape

video = cv2.VideoCapture(0)
while True:
    frame = video.read()[1]
    # frame = cv2.imread("cards/testing.jpg")
    card_imgs = find_cards.get_card_imgs(frame, 2)
    # try:
    #     card_identify.show_cards(card_imgs)
    # except:
    #     print("bad_image")
    cv2.imshow("og", frame)
    if len(card_imgs) > 0:
        # frame = cv2.imread("cards/2_oval_line.jpg")
        try:
            contours = get_shape_contours(card_imgs[0])
            cv2.drawContours(card_imgs[0], contours, -1, (0,255,0), 5)
            print(get_count(get_shape_contours(card_imgs[0])))
            print(get_shape_type(contours[0]))
            cv2.imshow("success", card_imgs[0])
        except:
            print("bad")
            #  cv2.imshow("failed", card_imgs[0])

    if cv2.waitKey(60) & 0xFF == ord('q'):
            break
    

    
    

