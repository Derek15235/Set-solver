import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import find_cards

class Card:
    def __init__(self, card_img):
        self.img_size = card_img.shape

        self.card_img = self.compress(card_img)
        self.shape_contours = self.get_shape_contours(self.card_img)

        self.count = len(self.shape_contours)
        self.shape = self.get_shape_type(self.shape_contours[0])
        self.color = self.get_color()

        

    def get_shape_contours(self, card):
        """
        INFO HERE
        :param [parameters]: 1D List of [R,G,B] values for each pixel
        :return: 
        """
        gray = cv2.cvtColor(card,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(1,1),1000)
        flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_OTSU)

        # In the orignal thresh, the shapes were black and the outside was white, but we flip it so the only thing outlining in our contours are the shapes, not the border of the image itself.
        thresh = cv2.bitwise_not(thresh)

        # Contours are sets of corner coordinates
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea,reverse=True)[:3]

        # Make sure all shapes are of similar areas, getting rid of small areas found
        final_shapes = [contours[0]]
        if len(contours) > 1:
            big_area = cv2.contourArea(contours[0])
            for i in range(1, len(contours)):
                if cv2.contourArea(contours[i]) / big_area > .8:
                    final_shapes.append(contours[i])

        return final_shapes

    def get_shape_type(self, image_shape):
        """
        INFO HERE
        :param [parameters]: 1D List of [R,G,B] values for each pixel
        :return: 
        """
        shapes = ["diamond", "oval", "wave"]
        shape_contours = {}

        for shape in shapes:
            shape_img = cv2.imread("cards/" + shape + ".jpg")
            shape_contours[shape] = self.get_shape_contours(shape_img)[0]

        best_match = .2
        best_shape = "No Match"

        for shape in shape_contours:
            match_score = cv2.matchShapes(image_shape, shape_contours[shape], 1, 0.0)

            if match_score < best_match:
                best_match = match_score
                best_shape = shape

        return best_shape

    def compress (self, img):
        """
        INFO HERE
        :param [parameters]: 1D List of [R,G,B] values for each pixel
        :return: 
        """
        # https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html
        Z = img.reshape((-1,3))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))

        return res2
    
    def get_color(self):
        # Ask if it is BGR
        unique_colors = np.unique(self.card_img.reshape(-1,3), axis=0)
        red = (48, 62, 147)
        green = (46, 78, 23)
        purple = (84, 33, 54)
        white = (0,0,0)
        for color in unique_colors:
            matches = {}
            matches["red"] = self.color_comparison_score(red, color)
            matches["green"] = self.color_comparison_score(green, color)
            matches["purple"] = self.color_comparison_score(purple, color)
            matches["white"] = self.color_comparison_score(white, color)
            best_match = min([matches["red"], matches["green"], matches["purple"], matches["white"]])
            # print("purple: " + str(matches["purple"]) + " green: " + str(matches["green"]))
            print(color)
            if best_match != matches["white"]:
                if best_match == matches["red"]:
                    return "red"
                elif best_match == matches["green"]:
                    return "green"
                elif best_match == matches["purple"]:
                    return "purple"



    def color_comparison_score(self, pixel1, pixel2):
        red_diff = abs(pixel1[0] - pixel2[0])
        green_diff = abs(pixel1[1]- pixel2[1])
        blue_diff = abs(pixel1[2] - pixel2[2])

        return red_diff + green_diff + blue_diff
            
        

video = cv2.VideoCapture(0)
while True:
    frame = video.read()[1]
    # frame = cv2.imread("cards/3_diamond_blank.jpg")
    card_imgs = find_cards.get_card_imgs(frame, 1)
    if len(card_imgs) > 0:
        card = Card(card_imgs[0])
        print(card.count)
        print(card.shape)
        print(card.color)
# while True:
        cv2.drawContours(card.card_img, card.shape_contours, -1, (0,255,0), 3)
        cv2.imshow("Hello", card.card_img)
    # frame = video.read()[1]
    # # frame = cv2.imread("cards/testing.jpg")
    # card_imgs = find_cards.get_card_imgs(frame, 2)
    # # try:
    # #     card_identify.show_cards(card_imgs)
    # # except:
    # #     print("bad_image")
    # cv2.imshow("og", frame)
    # if len(card_imgs) > 0:
    #     # frame = cv2.imread("cards/2_oval_line.jpg")
    #     try:
    #         card = Card(card_imgs[0])
    #         print(card.count)
    #         print(card.shape)
    #     except:
    #         print("bad")
    #         #  cv2.imshow("failed", card_imgs[0])

    if cv2.waitKey(60) & 0xFF == ord('q'):
            break
    

    
    

