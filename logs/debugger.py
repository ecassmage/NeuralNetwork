class Debug:
    def __init__(self, on=False):
        self.on = on

    def switch(self, switch):
        self.on = switch
        return self

    def print(self, message, end='\n'):
        if self.on:
            print(message, end=end)
