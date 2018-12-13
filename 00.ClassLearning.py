class Animals:
    def breathe(self):
        print('breathing')
    def move(self):
        print('moving')    
    def eat(self):
        print('eating')
class Mammals(Animals):
    def breastfeed(self):
        print('feeding young')
class Cats(Mammals):
    def __init__(self, spots = 4):
        self.spots = spots
    def catch_mouse(self):
        print('amd yes')

Kitty = Cats(10)
print(Kitty.spots)
Kitty.catch_mouse()
Kitty.breathe()
Kitty.breastfeed()
