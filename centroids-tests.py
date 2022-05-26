import pickle
import numpy as np
from enum import Enum
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.linalg import toeplitz
from utils import load_classes_from_folder, load_wav
from AkQuantization.LSPTOLSF import lpc_to_lsf, lsf_to_lpc
from process_signal import autocorrelation, calculate_lsp, get_energy, \
    get_edges, run_whole_signal, in_region,\
    euclidian_distance, dtw, get_new_matrix, \
    get_global_distance
from centroid import get_centroids

pf = 146
ws = 80
wa = ws
p = 16

k1 = .0001
k2 = .0003

gender = "male"
train_folder = "./corpus_digitos/training-examples/" + gender
test_folder = "./corpus_digitos/test-examples/" + gender

trains = load_classes_from_folder(train_folder, ".wav"); trains = np.array(trains)

centroids = get_centroids(trains, ws, wa, pf, k1, k2, p)

# Write the centroids to a pickle file
with open('centroids/centroids.pickle', 'wb') as handle:
    pickle.dump(centroids, handle, protocol=pickle.HIGHEST_PROTOCOL)

