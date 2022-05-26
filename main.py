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

sinal_treino = "training-examples/FAC_1A.wav"
sinal_teste = "FDC_1A.wav"

gender = "male"
train_folder = "./corpus_digitos/training-examples/" + gender
test_folder = "./corpus_digitos/test-examples/" + gender

trains = load_classes_from_folder(train_folder); trains = np.array(trains)
tests = load_classes_from_folder(test_folder); tests = np.array(tests)


# test_signal, _, _ = load_wav(test_folder + sinal_teste)

classes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
confusion_matrix = np.zeros((len(classes), len(classes)))
test_c = 0
for test_class in tests:
    # print("Classe -", classes[test_c])

    for test in test_class:
        test_signal = test[0]

        train_c = 0
        distances = np.ones_like(classes) * np.inf
        for train_class in trains:
            # print("Classe -", classes[train_c])
            # print(train_class.shape)

            lowest_global_distance = np.inf
            for train in train_class:
                train_signal = train[0]
                lsfs_train, energies_train, potency_train = run_whole_signal(train_signal, ws, wa, pf, k1, k2, p, to_plot=False)
                lsfs_test, energies_test, potency_test = run_whole_signal(test_signal, ws, wa, pf, k1, k2, p, to_plot=False)

                dtw_matrix = dtw(lsfs_train, lsfs_test, p, to_plot=False)

                min_matrix = get_new_matrix(dtw_matrix, to_plot=False)

                global_distance, new_matrix = get_global_distance(min_matrix, to_plot=False)

                if global_distance < lowest_global_distance:
                    lowest_global_distance = global_distance

            distances[train_c] = lowest_global_distance
            train_c += 1

        # print("Distances:", distances)
        predicted_class = classes[np.argmin(distances)]

        print("True Class:", classes[test_c])
        print("Predicted Class:", predicted_class)

        confusion_matrix[classes[test_c] - 1, predicted_class-1] += 1

    test_c += 1

print("Confusion Matrix")
print(confusion_matrix)

# Get accuracy from confusion matrix
accuracy = np.trace(confusion_matrix)/np.sum(confusion_matrix)
print("Accuracy:", accuracy)


# lsfs_train, energies_train, potency_train = run_whole_signal(train_results[0], ws, wa, pf, k1, k2, p, to_plot=False)
# lsfs_test, energies_test, potency_test = run_whole_signal(test_signal, ws, wa, pf, k1, k2, p, to_plot=False)
# dtw_matrix = dtw(lsfs_train, lsfs_test, p, to_plot=False)
# min_matrix = get_new_matrix(dtw_matrix, to_plot=True)
# global_distance, new_matrix = get_global_distance(min_matrix, to_plot=True)



# rate_train, sinal_train = read("./corpus_digitos/" + sinal_treino)
# max_train = abs(max(sinal_train)); sinal_train = sinal_train / max_train
#
# rate_test, sinal_test = read("./corpus_digitos/" + sinal_teste); max_test = abs(max(sinal_test))
# sinal_test = sinal_test / max_test
#
# lsfs_train, energies_train, potency_train = run_whole_signal(sinal_train, ws, wa, pf, k1, k2, p, to_plot=False)
# lsfs_test, energies_test, potency_test = run_whole_signal(sinal_test, ws, wa, pf, k1, k2, p, to_plot=False)
#
# print("lsfs_train: ", lsfs_train.shape)
# print("lsfs_test: ", lsfs_test.shape)
# distances_matrix = dtw(lsfs_train, lsfs_test, p, to_plot=False)
#
# min_matrix = get_new_matrix(distances_matrix, to_plot=False)
#
# global_distance, path = get_global_distance(min_matrix, to_plot=True)
# print(global_distance)