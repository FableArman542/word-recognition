import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_classes_from_folder, load_class_from_wav
from process_signal import autocorrelation, calculate_lsp, get_energy, \
    get_edges, run_whole_signal, in_region,\
    euclidian_distance, dtw, get_new_matrix, \
    get_global_distance

pf = 146
ws = 80
wa = ws
p = 16
tol = 1.2

k1 = .0001
k2 = .0003

gender = "male"
train_folder = "./corpus_digitos/training-examples/" + gender
test_folder = "./corpus_digitos/test-examples/" + gender

classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

centroids = pd.read_pickle(r'centroids/centroids.pickle')
centroid_distances = pd.read_pickle(r'centroids/centroids-max-distances.pickle')

tests = load_classes_from_folder(test_folder, extension=".wav"); tests = np.array(tests)

confusion_matrix = np.zeros((len(classes), len(classes)))

test_c = 0
for test_class in tests:

    for test in test_class:
        test_signal = test[0]
        # Calculate the lsfs
        lsfs_test, _, _ = run_whole_signal(test_signal, ws, wa, pf, k1, k2, p, to_plot=False)

        # Go through each centroid and calculate the distance
        distances = []
        for key in centroids:

            # Get the lsf of the centroid
            lsf_centroid = centroids[key]

            # Calculate the distance between the two signals
            dtw_matrix = dtw(lsf_centroid, lsfs_test, p, to_plot=False)
            min_matrix = get_new_matrix(dtw_matrix, to_plot=False)
            distance, new_matrix = get_global_distance(min_matrix, to_plot=False)
            distances = np.append(distances, distance)

        # Get the indexes of the 2 smallest distances
        min_indexes = np.argsort(distances)[:2]

        # Get the classes of the 2 smallest distances
        min_classes = [classes[i] for i in min_indexes]

        classes_ = np.array([])
        new_distances = np.array([])
        for nearest_class in min_classes:
            if nearest_class == 10:
                ajuda = load_class_from_wav(train_folder, 'O')
            elif nearest_class == 11:
                ajuda = load_class_from_wav(train_folder, 'Z')
            else:
                ajuda = load_class_from_wav(train_folder, str(nearest_class))

            for treino in ajuda:
                treino_signal = treino[0]

                # Calculate the lsfs
                lsfs_treino, _, _ = run_whole_signal(treino_signal, ws, wa, pf, k1, k2, p, to_plot=False)

                # Calculate the distance between the two signals
                dtw_matrix = dtw(lsfs_treino, lsfs_test, p, to_plot=False)
                min_matrix = get_new_matrix(dtw_matrix, to_plot=False)
                d, new_matrix = get_global_distance(min_matrix, to_plot=False)
                new_distances = np.append(new_distances, d)
                classes_ = np.append(classes_, nearest_class)

        idx = np.argmin(new_distances)
        predicted_class = classes_[idx]
        # Check if it is in the lexicon
        max_distance = centroid_distances[predicted_class-1]
        if new_distances[idx] < max_distance:
            confusion_matrix[classes[test_c] - 1, int(predicted_class) - 1] += 1
        else:
            confusion_matrix[classes[test_c] - 1, classes[len(classes)-1] - 1] += 1



    test_c += 1

print(confusion_matrix)
# Get accuracy from confusion matrix
accuracy = np.trace(confusion_matrix)/np.sum(confusion_matrix)
print("Accuracy:", accuracy)