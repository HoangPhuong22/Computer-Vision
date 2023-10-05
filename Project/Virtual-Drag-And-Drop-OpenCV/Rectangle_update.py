class Rectangle():
    def __init__(self, center, size = [200, 200]):
        self.center = center
        self.size = size
    def update(self, cursor):
        cx,cy = self.center
        w_rec, h_rec = self.size
        if (cx - w_rec // 2) <= cursor[0] <= (cx + w_rec // 2) and (cy - h_rec // 2) <= cursor[1] <= (cy + h_rec // 2):
            self.center = cursor
    def getCenter(self):
        return self.center
    def getSize(self):
        return self.size