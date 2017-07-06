class Door():

    def __init__(self, room1 = False, room2= False):

        self.state = False
        self.room1 = room1
        self.room2 = room2

    def open(self):
        self.state = True

    def close(self):
        self.state = False