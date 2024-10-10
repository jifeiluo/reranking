import os
import time
import math
import pickle
import shutil
import datetime


def load_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def save_pickle(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=4)