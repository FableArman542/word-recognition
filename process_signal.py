import numpy as np
from enum import Enum
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.linalg import toeplitz
from AkQuantization.LSPTOLSF import lpc_to_lsf, lsf_to_lpc

def autocorrelation(s, pf):
    r = np.zeros(pf)

    for k in range(pf):
        rk = 0
        for n in range(k, pf):
            if n < len(s):
                rk += s[n] * s[n - k]
        r[k] = rk

    return r / r[0]


def calculate_lsp(r, p):
    m = toeplitz(r[:p])
    mr = np.array(r[1:p + 1]) * -1
    minv = linalg.inv(m)
    a = np.dot(mr, minv)

    # Converting to lsp
    a = np.concatenate(([1.], a))
    lsf = lpc_to_lsf(a)  # * 0.5 / np.pi

    return lsf


class EdgeState(Enum):
    K1 = 0
    K2 = 1


def get_energy(s, pf):
    e = 0
    for n in range(pf):
        if n < len(s):
            e += s[n] * s[n]
    return e


def get_edges(potency, k1, k2, debug=False):
    state = EdgeState.K1
    begin = None
    end = None
    for i in range(len(potency)):
        if begin is not None and end is not None:
            break

        if begin is None and potency[i] >= k1 and state == EdgeState.K1:
            if debug: print("[INICIO] Encontrou K1")
            state = EdgeState.K2
        elif begin is None and potency[i] <= k1 and state == EdgeState.K2:
            if debug: print("[INICIO] Voltou a tás")
            state = EdgeState.K1
        elif begin is None and potency[i] >= k2 and state == EdgeState.K2:
            if debug: print("[INICIO] Encontrou k2 - Início")
            begin = i
        elif begin is not None and state == EdgeState.K2 and potency[i] <= k2:
            if debug: print("[FIM] Encontrou K2")
            state = EdgeState.K1
        elif begin is not None and state == EdgeState.K1 and potency[i] >= k2:
            if debug: print("[FIM] Voltou a trás")
            state = EdgeState.K2
        elif begin is not None and state == EdgeState.K1 and potency[i] <= k1:
            if debug: print("[FIM] Encontrou k1 = Fim")
            end = i

    return begin, end


def run_whole_signal(_signal, ws, wa, pf, k1, k2, p, to_plot=False):
    energies = np.array([])
    lsfs = np.zeros((int(len(_signal) / ws), p)).astype(np.float64)

    for i in range(0, int(len(_signal) / ws)):
        ni = i * ws
        window = _signal[ni:ni + wa]

        energy = get_energy(window, pf)
        energies = np.append(energies, energy)

        r = autocorrelation(window, pf)

        lsf = calculate_lsp(r, p)
        lsfs[i] = lsf

    potency = energies / ws  # Passar para potencia

    begin, end = get_edges(potency, k1, k2)  # Remover silencio

    cut_potency = potency[begin:end]
    cut_lsfs = lsfs[begin:end]

    if to_plot:
        plt.title("Signal")
        plt.plot(_signal)
        plt.show()

        plt.title("Cut Potency")
        plt.plot(potency[begin:end])
        plt.show()

    return cut_lsfs, energies, cut_potency


def in_region(i, j, rows, columns):
    J = columns
    I = rows

    A = j <= .5 * i - .5 * I + J - .5
    B = j >= 2 * i - 2 * I + J + 1
    C = j >= .5 * i
    D = j <= 2 * i

    return A and B and C and D


def euclidian_distance(lsf_train, lsf_test, p, wn=1):
    aux = 0
    for n in range(p):
        aux += wn * ((lsf_train[n] - lsf_test[n]) ** 2)
    return np.sqrt(aux)


def dtw(lsfs_train, lsfs_test, p, to_plot=False):
    matrix = np.zeros((len(lsfs_test), len(lsfs_train)))

    rows = matrix.shape[0]
    columns = matrix.shape[1]

    for i in range(len(lsfs_test)):
        for j in range(len(lsfs_train)):
            if in_region(i, j, rows, columns):
                matrix[i, j] = euclidian_distance(lsfs_train[j], lsfs_test[i], p)
            else:
                matrix[i, j] = np.inf

    if to_plot:
        plt.matshow(matrix)
        plt.colorbar()
        plt.show()

    return matrix


def get_new_matrix(dtw_matrix, to_plot=False):
    new_matrix = np.zeros_like(dtw_matrix)
    for i in range(len(dtw_matrix)):
        for j in range(len(dtw_matrix[0])):
            if i == 0 and j == 0:
                new_matrix[i, j] = dtw_matrix[i, j]
            elif i == 0:
                new_matrix[i, j] = dtw_matrix[i, j] + new_matrix[i, j - 1]
            elif j == 0:
                new_matrix[i, j] = dtw_matrix[i, j] + new_matrix[i - 1, j]
            else:
                new_matrix[i, j] = dtw_matrix[i, j] + min(new_matrix[i - 1, j], new_matrix[i, j - 1],
                                                          new_matrix[i - 1, j - 1])

    if to_plot:
        plt.matshow(new_matrix)
        plt.colorbar()
        plt.show()

    return new_matrix


def get_global_distance(min_matrix, to_plot=False):
    I = len(min_matrix)
    J = len(min_matrix[0])

    # Start in min_matrix[I - 1, J - 1] and follow the path of the lowest cost
    i = I - 1
    j = J - 1
    new_matrix = np.zeros_like(min_matrix)
    global_distance = min_matrix[i, j]
    # new_matrix[i, j] = 1
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        elif min_matrix[i - 1, j] < min_matrix[i, j - 1] and min_matrix[i - 1, j] < min_matrix[i - 1, j - 1]:
            i -= 1
        elif min_matrix[i, j - 1] < min_matrix[i - 1, j] and min_matrix[i, j - 1] < min_matrix[i - 1, j - 1]:
            j -= 1
        else:
            i -= 1
            j -= 1
        new_matrix[i, j] = 1

    if to_plot:
        plt.matshow(new_matrix)
        plt.show()
        print("DISTANCE", global_distance)
        print("SOMA", np.sum(new_matrix == 1))
    return global_distance / np.sum(new_matrix == 1), new_matrix