

class Average:

    def __init__(self):
        self.value = 0
        self.n = 0

    def reset(self):
        self.value = 0
        self.n = 0

    def add(self, x):
        self.value += x
        self.n += 1

    def get(self):
        return self.value / max(1, self.n)
