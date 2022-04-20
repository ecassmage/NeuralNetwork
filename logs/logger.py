import time


class Log:
    def __init__(self, file):
        self.file = open(file, 'w')
        self.X, self.y = [], []
        self.start = time.time()

        self.extra_information = {}

    def write_loss(self, net):
        pass

    def add(self, key, information):
        self.extra_information[key] = self.extra_information.get(key, []) + (information if isinstance(information, list) else [information])

    def write_extra(self, message=None, key=None, function=None, args=(), end='\n'):
        self.write(message + (self.extra_information[key] if not function else function(self.extra_information[key], *args)), end)

    def write(self, message: str, end='\n'):
        self.file.write(message + end)

    def add_to_plot(self, X, y):
        self.X += X
        self.y += y

    def close(self):
        self.write(str(self.X))
        self.write(str(self.y))
        self.write(f"Time Total >> {time.time() - self.start}")
        self.file.close()
