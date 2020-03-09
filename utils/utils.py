import pickle


def load_class(loadpath):
    file = open(loadpath, 'rb')
    class_ = pickle.load(file)
    file.close()
    return class_
