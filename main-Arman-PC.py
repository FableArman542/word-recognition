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

pf = 146
ws = 80
wa = ws
p = 16

k1 = .0001
k2 = .0003

sinal_treino = "training-examples/female/FAC_1A.wav"
sinal_teste = "test-examples/female/FDC_1A.wav"

rate_train, sinal_train = read("./corpus_digitos/" + sinal_treino)
max_train = abs(max(sinal_train)); sinal_train = sinal_train / max_train

rate_test, sinal_test = read("./corpus_digitos/" + sinal_teste); max_test = abs(max(sinal_test))
sinal_test = sinal_test / max_test

lsfs_train, energies_train, potency_train = run_whole_signal(sinal_train, ws, wa, pf, k1, k2, p, to_plot=False)
lsfs_test, energies_test, potency_test = run_whole_signal(sinal_test, ws, wa, pf, k1, k2, p, to_plot=False)

print("lsfs_train: ", lsfs_train.shape)
print("lsfs_test: ", lsfs_test.shape)
distances_matrix = dtw(lsfs_train, lsfs_test, p, to_plot=False)

min_matrix = get_new_matrix(distances_matrix, to_plot=True)

global_distance, path = get_global_distance(min_matrix, to_plot=True)
print(global_distance)