import numpy as np
from enum import Enum
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.linalg import toeplitz
from AkQuantization.codeLSP import codeLSP
from AkQuantization.decode_lsp import decodeLSP
from AkQuantization.LSPTOLSF import lpc_to_lsf, lsf_to_lpc


def autocorrelation(s, pf):
    r = np.zeros(pf)

    for k in range(pf):
        rk = 0
        for n in range(k, pf):
            if n < len(s):
                rk += s[n] * s[n - k]
        r[k] = rk

    return r/r[0]

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

    potency = energies / wa  # Passar para potencia

    begin, end = get_edges(potency, k1, k2)  # Remover silencio

    if to_plot:

        plt.title("Signal")
        plt.plot(_signal)
        plt.show()

        plt.title("Cut Potency")
        plt.plot(potency[begin:end])
        plt.show()


pf = 146
ws = 80
wa = ws
p = 16

k1 = .01
k2 = .02
sinal_treino = "training-examples/FAC_1A.wav"
sinal_teste = "test-examples/FDC_1A.wav"

ss = read("./corpus_digitos/" + sinal_teste)
sinal = ss[1]
rate = ss[0]
maximo = abs(max(sinal))
sinal = sinal / maximo  # Normalize signal

run_whole_signal(sinal, ws, wa, pf, k1, k2, p, to_plot=True)
