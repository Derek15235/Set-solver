import cv2
import numpy as np

class Card:
    def __init__(self, card_img, contour):
        self.contour = contour
        self.img_size = card_img.shape

        self.card_img = card_img
        self.shape_contours = self.get_shape_contours(self.card_img)
        self.shape_img = self.make_shape_img(card_img, self.shape_contours[0])

        self.count = len(self.shape_contours)
        self.shape = self.get_shape_type(self.shape_contours[0])
        self.color = self.get_color()
        self.fill = self.get_fill()

        

    def get_shape_contours(self, card):
        """
        INFO HERE
        :param [parameters]:
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
        Takes in the cropped shape image and compares it to each of the three shapes
        :param [image_shape]: 
        :return: 
        """
        shapes = ["Diamond", "Oval", "Wave"]
        shape_contours = {}

        for shape in shapes:
            shape_img = cv2.imread("cards/" + shape + ".jpg")
            shape_contours[shape] = self.get_shape_contours(shape_img)[0]

        best_match = .2
        # The most abnormal shape is diamond, so if no shape matches, assume diamond
        best_shape = "Diamond"

        for shape in shape_contours:
            match_score = cv2.matchShapes(image_shape, shape_contours[shape], 1, 0.0)

            if match_score < best_match:
                best_match = match_score
                best_shape = shape

        return best_shape

    def compress (self, img):
        """
        INFO HERE
        :param [parameters]:
        :return: 
        """
        # Start of code from: https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html
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
        # End of code from: https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html

        return res2
    
    def make_shape_img(self, og_image, shape_contour):
        # Start of code from: https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/
        # rotate img
        rect = cv2.minAreaRect(shape_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        cv2.drawContours(og_image, [box], 0, (255, 255, 255), 0)

        # get width and height of the detected rectangle
        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        # coordinate of the points in box points after the rectangle has been
        # straightened
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")

        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv2.warpPerspective(og_image, M, (width, height))
        # End of code from: https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/

        warped = cv2.resize(warped, (500,300))

        return warped


    
    def get_color(self):
        simple_shape = self.compress(self.shape_img)
        unique_colors = np.unique(simple_shape.reshape(-1,3), axis=0)

        BGR_color = unique_colors[0]

        red_ratio = float(BGR_color[2]) / float(BGR_color[0] + BGR_color[1] + BGR_color[2])
        blue_ratio = float(BGR_color[0]) / float(BGR_color[0] + BGR_color[1] + BGR_color[2])
        green_ratio = float(BGR_color[1]) / float(BGR_color[0] + BGR_color[1] + BGR_color[2])
        if green_ratio > red_ratio and green_ratio > blue_ratio:
            return "Green"
        if red_ratio > blue_ratio * 1.25:
            return "Red"
        else:
            return "Purple"

        
    def get_fill(self):
        gray = cv2.cvtColor(self.shape_img,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(1,1),1000)
        flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_OTSU)

        # Crop
        w = 100
        h = 100
        y = int(thresh.shape[0]/2 - h/2)
        x = int(thresh.shape[1]/2 - w/2)
        cropped = thresh[y:y+h, x:x+w]

        # Counting black and white pixels then finding proportion of black
        unique_colors = np.unique(cropped.reshape(-1,1), axis=0)
        
        if len(unique_colors) == 1:
            if unique_colors[0][0] == 255:
                # Double check to make sure actually empty, not striped
                # Crop shape image (no augments)
                w = 100
                h = 100
                y = int(self.shape_img.shape[0]/2 - h/2)
                x = int(self.shape_img.shape[1]/2 - w/2)
                cropped = self.shape_img[y:y+h, x:x+w]

                # Take sample out of the white part of the card
                white_corner = self.card_img[0:100, 0:100]
                mean_white = np.average(white_corner, axis=(0,1))
                mean_fill = np.average(cropped, axis=(0,1))
                if self.color_difference(mean_white, mean_fill) > 20:
                    return "Striped"
                else:
                    return "Empty"
                # return "Empty"
            else:
                return "Solid"
        else:
            return "Striped"
        
    def color_difference(self, color1, color2):
        blue_diff = abs(color1[0] - color2[0])
        green_diff = abs(color1[1] - color2[1])
        red_diff = abs(color1[2] - color2[2])

        return blue_diff + green_diff + red_diff
    
    def __str__(self):
        return str(self.count) + " " + self.color + " " + self.shape + " " + str(self.fill)

    
    

