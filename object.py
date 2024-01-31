from abc import ABC, abstractmethod

class GameObject(ABC):
    def __init__(self, x, y, width, height, game):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.game = game
        self.gc = game.gc

    @abstractmethod
    def draw(self):
        pass

    @abstractmethod
    def update(self):
        pass

    def get_mid(self):
        return (self.x + self.width / 2, self.y + self.height / 2)

    def collides(self, object):
        mid = self.get_mid()
        obj_mid = object.get_mid()

        return abs(mid[0] - obj_mid[0]) < self.width / 2 + object.width / 2 \
        and abs(mid[1] - obj_mid[1]) < self.height / 2 + object.height / 2
    
    def move(self, mx, my):
        self.x += mx
        self.y += my
