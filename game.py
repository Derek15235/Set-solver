import cv2
import numpy as np 
from card import Card

class Game:
    def __init__(self, video, num_cards=12):
        self.video = video
        self.num_cards = num_cards
        self.frame = self.video.read()[1]

    def get_contours(self):
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
            approx.append(cv2.approxPolyDP(contour, .1 * peri, True))

        return approx

    def get_card_imgs(self):
        """
        INFO HERE
        :param [parameters]: 1D List of [R,G,B] values for each pixel
        :return: 
        """
        approx = self.get_contours()

        individual_card_images = []

        # Make a new image of everything masked except the individual cards
        for i in range(len(approx)):
            # Make sure rectangular and aren't too small
            if len(approx[i]) == 4 and cv2.contourArea(approx[i]) > 2000:
                # mask = np.zeros((frame.shape[0], frame.shape[1]), dtype="uint8")
                # final_mask = cv2.drawContours(mask, [approx[i]], -1, (255,255,255), -1)
                if i < len(individual_card_images) - 1:
                    # individual_card_images[i] = cv2.bitwise_and(frame, frame, mask=final_mask)
                    individual_card_images[i] = self.crop(self.frame, approx[i])
                else:
                    # individual_card_images.append(cv2.bitwise_and(frame, frame, mask=final_mask))
                    individual_card_images.append(self.crop(self.frame, approx[i]))
        
        return individual_card_images

    def crop(self, original_image, contour):
        """
        INFO HERE
        :param [parameters]: 1D List of [R,G,B] values for each pixel
        :return: 
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
        cropped = masked_img[y+15:y+h-15, x+15:x+w-15]
        return cropped
    
    def run(self):
        while True:
            key = cv2.waitKey(60) & 0xFF
            if key == ord('r'):
                self.frame = self.video.read()[1]

                card_imgs = self.get_card_imgs()
                frame_contours = self.get_contours()[:len(card_imgs)]

                cv2.drawContours(self.frame, frame_contours, -1, (0,255,0), 5)

                for i in range(len(card_imgs)):
                    card = Card(card_imgs[i])
                    print(f"Card {i+1}")
                    print(card.count)
                    print(card.shape)
                    print(card.color)
                    print(card.fill)
    
            cv2.imshow("Game", self.frame)

            if key == ord('q'):
                    break
                    
    
 


video = cv2.VideoCapture(0)
frame = video.read()[1]
game = Game(video)
game.run()