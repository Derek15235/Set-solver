from card import Card
from scanner import Scanner
import cv2

class Game:
    def __init__(self, mode=0):
        self.mode = mode

    def find_set(self):
        for i in range(len(self.cards)):
            for j in range (i+1, len(self.cards)):
                # Find needed shape
                if self.cards[i].shape == self.cards[j].shape:
                    needed_shape = self.cards[i].shape
                else:
                    shapes = ["Diamond", "Oval", "Wave"]
                    shapes.remove(self.cards[i].shape)
                    shapes.remove(self.cards[j].shape)
                    needed_shape = shapes[0]
                
                # Find needed count
                if self.cards[i].count == self.cards[j].count:
                    needed_count = self.cards[i].count
                else:
                    numbers = [1, 2, 3]
                    numbers.remove(self.cards[i].count)
                    numbers.remove(self.cards[j].count)
                    needed_count = numbers[0] 

                # Find needed fill
                if self.cards[i].fill == self.cards[j].fill:
                    needed_fill = self.cards[i].fill
                else:
                    fills = ["Empty", "Solid", "Striped"]
                    fills.remove(self.cards[i].fill)
                    fills.remove(self.cards[j].fill)
                    needed_fill = fills[0]

                # Find needed color
                if self.cards[i].color == self.cards[j].color:
                    needed_color = self.cards[i].color
                else:
                    colors = ["Red", "Purple", "Green"]
                    colors.remove(self.cards[i].color)
                    colors.remove(self.cards[j].color)
                    needed_color = colors[0]

                for card in self.cards:
                    if card.shape == needed_shape and card.count == needed_count and card.fill == needed_fill and card.color == needed_color:
                        return [self.cards[i], self.cards[j], card]
        return []

    def run(self):
        if self.mode == 0:
            video = cv2.VideoCapture(0)
            frame = video.read()[1]
        else:
            frame = cv2.imread("cards/testing.jpg")

        while True:
            key = cv2.waitKey(60) & 0xFF
            if key == ord('r'):
                if self.mode == 0:
                    frame = video.read()[1]
                else:
                    frame = cv2.imread("cards/testing.jpg")
                scanner = Scanner(frame)

                card_imgs = scanner.get_card_imgs()
                card_contours = scanner.get_card_contours()[:len(card_imgs)]
                self.cards = []

                for i in range(len(card_imgs)):
                    card = Card(card_imgs[i],card_contours[i])
                    self.cards.append(card)
                    cv2.drawContours(frame, [card.contour], -1, (0,0,255), 5)
                    print(card)

                set = self.find_set()
                if len(set) == 0:
                    print("No Sets Found")
                else:
                    for card in set:
                        cv2.drawContours(frame, [card.contour], -1, (0,255,0), 5)
                        x,y,w,h = cv2.boundingRect(card.contour)
                        cv2.putText(frame, "Set", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.imshow("Game", frame)

            if key == ord('q'):
                    break

if __name__ == '__main__':
    game = Game()
    game.run()