import math
import numpy as np
from process_signal import autocorrelation, calculate_lsp, get_energy, \
    get_edges, run_whole_signal, in_region,\
    euclidian_distance, dtw, get_new_matrix, \
    get_global_distance


def get_centroids(trains, ws, wa, pf, k1, k2, p):
    # 9 = O
    # 10 = Z
    classes = { 0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None, 10: None }

    class_number = 0
    for train_class in trains:
        # print("Class", class_number)
        signals = []
        distances_per_class = []
        # Each class has a list of signals
        for i in range(len(train_class)):
            instance = train_class[i]
            train_signal = instance[0]
            lsfs_train, energies_train, potency_train = run_whole_signal(train_signal, ws, wa, pf, k1, k2, p, to_plot=False)
            distances = np.array([])
            # Loop through every other signal in the class
            for j in range(len(train_class)):
                other_signal = train_class[j]
                if i != j:
                    lsfs_other, _, _ = run_whole_signal(other_signal[0], ws, wa, pf, k1, k2, p, to_plot=False)

                    # Calculate the distance between the two signals
                    dtw_matrix = dtw(lsfs_train, lsfs_other, p, to_plot=False)
                    min_matrix = get_new_matrix(dtw_matrix, to_plot=False)
                    distance, new_matrix = get_global_distance(min_matrix, to_plot=False)
                    # if not math.isinf(distance):
                    distances = np.append(distances, distance)
            # print("Distances", distances)
            # signals.append(instance)
            signals.append(lsfs_train)
            # print(lsfs_train.shape)
            distances_per_class.append(np.sum(distances))

        # Get the index of the signal with the lowest distance
        index = np.argmin(distances_per_class)

        # print(np.shape(signals))
        # print(np.shape(signals[index]))
        # Add the signal to the class
        classes[class_number] = signals[index]

        class_number += 1

    print("Centroids:")
    for key in classes:
        print(key, classes[key])

    return classes