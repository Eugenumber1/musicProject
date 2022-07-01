import processing

class Track:

    def __init__(self, path):
        self.path = path
        self.name = processing.name_retriever(path)

    genre = None

    def setGenre(self, genre):
        self.genre = genre

    def getGenre(self):
        return self.genre

