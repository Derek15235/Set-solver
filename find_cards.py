import cv2
import numpy as np 

def get_card_imgs(frame, num_cards):
    # Start of code from: https://arnab.org/blog/so-i-suck-24-automating-card-games-using-opencv-and-python/
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(1,1),1000)
    flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_OTSU)

    # Contours are sets of corner coordinates
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea,reverse=True)[:num_cards]
    # End of code from: https://arnab.org/blog/so-i-suck-24-automating-card-games-using-opencv-and-python/
    approx = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx.append(cv2.approxPolyDP(contour, .1 * peri, True))
    individual_card_images = []

    # Make a new image of everything masked except the individual cards
    for i in range(len(approx)):
        if len(approx[i]) == 4:
            # mask = np.zeros((frame.shape[0], frame.shape[1]), dtype="uint8")
            # final_mask = cv2.drawContours(mask, [approx[i]], -1, (255,255,255), -1)
            if i < len(individual_card_images) - 1:
                # individual_card_images[i] = cv2.bitwise_and(frame, frame, mask=final_mask)
                individual_card_images[i] = crop(frame, approx[i])
            else:
                # individual_card_images.append(cv2.bitwise_and(frame, frame, mask=final_mask))
                individual_card_images.append(crop(frame, approx[i]))
    
    return individual_card_images

def crop(original_image, contour):
    # Mask
    mask = np.zeros((original_image.shape[0], original_image.shape[1]), dtype="uint8")
    final_mask = cv2.drawContours(mask, [contour], -1, (255,255,255), -1)
    masked_img = cv2.bitwise_and(original_image, original_image, mask=final_mask)

    # Changed masked part (all black pixels) to be white
    hsv = cv2.cvtColor(masked_img,cv2.COLOR_BGR2HSV)
    black_pixels = cv2.inRange(hsv,np.array([0,0,0]),np.array([0,0,0]))
    masked_img[black_pixels>0] = (255,255,255)


    # Find bounding box
    x,y,w,h = cv2.boundingRect(contour)

    # Return only the content inside of the bounding box
    cropped = masked_img[y+15:y+h-15, x+15:x+w-15]
    return cropped


def show_cards(individual_card_images):
    for i in range(len(individual_card_images)):
        cv2.imshow(f"Card {i+1}", individual_card_images[i])




