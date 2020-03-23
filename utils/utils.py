import torch


def load_class(loadpath):
    if torch.cuda.is_available():
        return torch.load(loadpath)
    return torch.load(loadpath, map_location=torch.device('cpu'))
