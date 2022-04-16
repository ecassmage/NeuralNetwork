import os


class Log:
    def __init__(self, name="log.txt"):
        path = os.getcwd() + '\\logs\\'
        self.file = open(path + name, 'w')

    def write(self, message='', end='\n'):
        self.file.write(message + end)

    def close(self):
        self.file.close()


def main():
    pass


if __name__ == '__main__':
    main()
    pass
