"""

Been using this for probably a year at this point, just changing it slightly whenever.

Name: Prisoners Dilemma, Exhaustive Search w/ basic neural network
Developer: Evan Morrison
Version 1.1.0
Since 1.0.0
"""


def getConfig(keepComments=False) -> dict:
    """
    This will open and collect the config file, storing it into a dictionary.
    :param keepComments: default False, will remove #COMMENT sections from json files if this is false, else it won't
    :return: returns the config.json file as a dictionary.
    """
    import json
    import os

    for file in os.listdir(os.path.abspath(os.curdir)):
        if len(file) > 5 and file[-5:] == ".json":
            filename = file
            break
    else:
        raise FileNotFoundError("a .json file was not located")
    try:
        file = open(filename)
    except FileNotFoundError:
        try:
            file = open(os.path.abspath(os.curdir) + '\\' + filename)
        except FileNotFoundError:
            file = open(os.path.dirname(os.path.abspath(os.curdir)) + '\\' + filename)

    jsonFile = json.load(file)
    file.close()
    __cleaningJSON(jsonFile, keepComments)
    return jsonFile


def __cleaningJSON(jsonFile, keepComments):
    if not keepComments:
        __removeComments(jsonFile)


def __removeComments(jsonFile):
    if type(jsonFile) is dict:
        __removeCommentsDict(jsonFile)
    else:
        __removeCommentsList(jsonFile)


def __removeCommentsDict(jsonFile: dict):
    string = "#COMMENT"
    for key in list(jsonFile.keys()):
        if type(key) is str and key[:len(string)] == string:
            jsonFile.pop(key)
        elif type(jsonFile[key]) is dict:
            __removeCommentsDict(jsonFile[key])
        elif type(jsonFile[key]) is list:
            __removeCommentsList(jsonFile[key])


def __removeCommentsList(jsonFile: list):
    string = "#COMMENT"
    for key in jsonFile:
        if type(key) is str and key[:len(string)] == string:
            jsonFile.pop(key)
        elif type(key) is dict:
            __removeCommentsDict(key)
        elif type(key) is list:
            __removeCommentsList(key)


if __name__ == '__main__':
    dictionary = getConfig()
    print(dictionary)
