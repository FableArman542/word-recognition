import math
import pickle
import numpy as np
import pandas as pd
from utils import load_classes_from_folder, load_class_from_wav
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

gender = "male"
train_folder = "./corpus_digitos/training-examples/" + gender
test_folder = "./corpus_digitos/test-examples/" + gender

centroids = pd.read_pickle(r'centroids/centroids.pickle')

classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

classes_dict = { 0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None, 10: None }

for key in centroids:
    # Get the lsf of the centroid
    lsf_centroid = centroids[key]

    if classes[key] == 10:
        ajuda = load_class_from_wav(train_folder, 'O')
    elif classes[key] == 11:
        ajuda = load_class_from_wav(train_folder, 'Z')
    else:
        ajuda = load_class_from_wav(train_folder, str(classes[key]))

    distances = np.array([])
    for train in ajuda:
        train_signal = train[0]

        # Calculate the lsfs
        lsfs_train, _, _ = run_whole_signal(train_signal, ws, wa, pf, k1, k2, p, to_plot=False)

        # centroid_max_distance = 0

        # Calculate the distance between the two signals
        dtw_matrix = dtw(lsf_centroid, lsfs_train, p, to_plot=False)
        min_matrix = get_new_matrix(dtw_matrix, to_plot=False)
        distance, new_matrix = get_global_distance(min_matrix, to_plot=False)
        if not math.isinf(distance):
            distances = np.append(distances, distance)

    classes_dict[key] = np.max(distances)


# Go throught the dictionary and print the classes
for key in classes_dict:
    print(key, classes_dict[key])

with open('centroids/centroids-max-distances.pickle', 'wb') as handle:
    pickle.dump(classes_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


