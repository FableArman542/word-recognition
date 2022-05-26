import os
import numpy as np
from re import search
from scipy.io.wavfile import read

def load_classes_from_folder(folder, extension='A.wav'):
    # classes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    class1 = []; class2 = []; class3 = []
    class4 = []; class5 = []; class6 = []
    class7 = []; class8 = []; class9 = []
    for filename in os.listdir(folder):
        if filename.endswith(extension):
            rate, sinal = read(os.path.join(folder, filename))
            maximum = abs(max(sinal))
            sinal = sinal / maximum
            if search('1', filename):
                class1.append((sinal, rate, maximum))
            elif search('2', filename):
                class2.append((sinal, rate, maximum))
            elif search('3', filename):
                class3.append((sinal, rate, maximum))
            elif search('4', filename):
                class4.append((sinal, rate, maximum))
            elif search('5', filename):
                class5.append((sinal, rate, maximum))
            elif search('6', filename):
                class6.append((sinal, rate, maximum))
            elif search('7', filename):
                class7.append((sinal, rate, maximum))
            elif search('8', filename):
                class8.append((sinal, rate, maximum))
            elif search('9', filename):
                class9.append((sinal, rate, maximum))

    return np.array(class1), np.array(class2), np.array(class3), np.array(class4), np.array(class5), np.array(class6), np.array(class7), np.array(class8), np.array(class9)

def load_wav(filename):
    rate, sinal = read(filename)
    maximum = abs(max(sinal))
    sinal = sinal / maximum
    return sinal, rate, maximum