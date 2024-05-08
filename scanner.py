import cv2
import numpy as np 
from card import Card

class Scanner:
    def __init__(self, frame, num_cards=12):
        """
        Initializes scanner object with the image to scan and number of cards to find
        :param frame: BGR image of the full image with the cards
        :param num_cards: number of cards to scan for
        """
        self.num_cards = num_cards
        self.frame = frame

    def get_card_contours(self):
        """
        Scan the given image for all the card outlines and returning the polygon version of each outline
        :return: a list of the polygon approximated outlines of each card
        """
        # Start of code from: https://arnab.org/blog/so-i-suck-24-automating-card-games-using-opencv-and-python/
        gray = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(1,1),1000)
        flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_OTSU)

        # Contours are sets of corner coordinates
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea,reverse=True)[:self.num_cards]
        # End of code from: https://arnab.org/blog/so-i-suck-24-automating-card-games-using-opencv-and-python/

        approx = []
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approximation = cv2.approxPolyDP(contour, .1 * peri, True)
            area = cv2.contourArea(approximation)
            print(area)
            if len(approximation) == 4 and area > 20000:
                print(cv2.contourArea(approximation))
                approx.append(approximation)  

        return approx

    def get_card_imgs(self):
        """
        Create a list of all the different images of each card
        :return: list of images of each individual cards
        """
        approx = self.get_card_contours()

        individual_card_images = []

        # Make a new image of everything masked except the individual cards
        for i in range(len(approx)):
            if i < len(individual_card_images) - 1:
                individual_card_images[i] = self.crop(self.frame, approx[i])
            else:
                individual_card_images.append(self.crop(self.frame, approx[i]))
        
        return individual_card_images

    def crop(self, original_image, contour):
        """
        Crop the frame to just include the card image
        :param original_image: a masked image of the entire frame where the only thing showing the specific card
        :param contour: the specfic card's outline
        :return: a rectangular cropped image of mainly jsut the card
        """
        # Mask
        # Start of code from: https://www.youtube.com/watch?v=f6VgWTD_7kc
        mask = np.zeros((original_image.shape[0], original_image.shape[1]), dtype="uint8")
        final_mask = cv2.drawContours(mask, [contour], -1, (255,255,255), -1)
        masked_img = cv2.bitwise_and(original_image, original_image, mask=final_mask)
        # End of code from: https://www.youtube.com/watch?v=f6VgWTD_7kc

        # Changed masked part (all black pixels) to be white
        hsv = cv2.cvtColor(masked_img,cv2.COLOR_BGR2HSV)
        black_pixels = cv2.inRange(hsv,np.array([0,0,0]),np.array([0,0,0]))
        masked_img[black_pixels>0] = (255,255,255)

        # Find bounding box
        x,y,w,h = cv2.boundingRect(contour)

        # Return only the content inside of the bounding box
        cropped = masked_img[y:y+h, x:x+w]
        return cropped
    
  




                    
    
 


