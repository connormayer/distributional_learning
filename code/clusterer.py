import argparse
import numpy as np

from math import exp, log, pi
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

DEFAULT_VARIABILITY_SCALAR = 1
DEFAULT_CONSTRAIN_PARTITIONS = True
DEFAULT_CONSTRAIN_PCS = True

VALUE_EXT = '.data'
SOUND_EXT = '.sounds'
CONTEXT_EXT = '.contexts'

def remove_duplicates(my_list):
    seen = set()
    seen_add = seen.add
    return [
        x for x in [tuple(y) for y in my_list] 
        if not (x in seen or seen_add(x))
    ]

def do_clustering(input_file_stem, output_file, v_scalar=DEFAULT_VARIABILITY_SCALAR, 
                  constrain_partition=False,
                  constrain_pcs=False):
    values = np.loadtxt(input_file_stem + VALUE_EXT)
    with open(input_file_stem + SOUND_EXT, 'r') as sound_file:
        sounds = sound_file.read().strip().split(' ')
    with open(input_file_stem + CONTEXT_EXT, 'r') as context_file:
        contexts = context_file.read().strip().split(' ')

    classes = [tuple(sounds)]
    classes.extend(find_classes(values, sounds, v_scalar, constrain_partition, constrain_pcs))
    classes = remove_duplicates(classes)

    print("Found classes:")

    with open(output_file, 'w') as f:
        for c in classes:
            print(c)
            print(' '.join(c), file=f)

def calculate_mean_and_variance(X, n):
    '''
    Calculate the mean and variance of a cluster
    '''
    my_sum = 0
    sumsq = 0

    sorted_X = sorted(X)

    median = sorted_X[len(sorted_X) // 2]
    for item in sorted_X:
        my_sum += item - median
        sumsq += (item - median) * (item - median)
    mean = my_sum / n + median

    if n > 1:
        variance = (sumsq - my_sum * my_sum / n) / (n - 1)
    else:
        variance = 0

    return mean, variance

def compute_bic(kmeans, X):
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    # number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)

    lamb = []
    coeff = []
    means = []
    sigmas = []

    for k in range(m):
        lamb.append(n[k] / len(labels))
        mean, variance = calculate_mean_and_variance(X[np.where(labels == k)], n[k])
        means.append(mean)
        sigmas.append(variance)

        # If we can't calculate variance within cluster, use minimum distance to point
        # in other cluster.
        if variance == 0 or n[k] == 1:
            X_sorted = np.copy(X)
            X_sorted.sort(axis=0)
            left = np.where(X_sorted == means[k])
            left[0][0] -= 1
            right = np.where(X_sorted == means[k])
            right[0][0] += 1

            if right[0][0] >= len(X):
                dmin = X_sorted[np.where(X_sorted==means[k])] - X_sorted[left]
            elif left[0][0] < 0:
                dmin = X_sorted[right] - X_sorted[np.where(X_sorted==means[k])]
            else:
                dmin = min(X_sorted[np.where(
                    X_sorted==means[k])] - X_sorted[left], 
                    X_sorted[right] - X_sorted[np.where(X_sorted==means[k])]
                )

            if variance == 0:
                sigmas[-1] = dmin * dmin / 4.0 / 9.0
            if n[k] == 1:
                sigmas[-1] = dmin * dmin

        coeff.append(lamb[k] / (2.0 * pi * sigmas[-1])**0.5)

    log_likelihood = 0
    for item in X:
        likelihood = 0
        for k in range(m):
            likelihood += coeff[k] * exp(-(item - means[k]) * (item - means[k]) / (2 * sigmas[k]))
        log_likelihood += log(likelihood)

    bic = 2 * log_likelihood - (3 * m - 1) * log(len(X))
    return bic

def find_classes(input_data, sounds, v_scalar=DEFAULT_VARIABILITY_SCALAR,
                 constrain_partition=False,
                 constrain_pcs=False,
                 visited_classes=None):

    full_classes_list = []

    if not visited_classes:
        visited_classes = []

    # Do PCA on the input data
    pca = PCA()
    pca_values = pca.fit_transform(input_data)

    if constrain_pcs:
        highest_dim = 1
    else:
        # If we're looking at all PCs, calculate which ones we will examine
        # based on scaled Kaiser's stopping criterion.
        mean_eig = np.mean(pca.explained_variance_) * v_scalar
        highest_dim = max(0, np.argmax(pca.explained_variance_ < mean_eig))

    if constrain_partition:
        # Only cluster into a maximum of two classes
        max_clusters = 2
    else:
        # Find no more than three clusters.
        max_clusters = min(3, len(sounds))

    # Go through all the PCS we want to cluster over
    for i in range(highest_dim):
        classes_list = []
        sub_classes = []
        col = pca_values[:, i]
        col_reshaped = col.reshape(-1, 1)

        # Do 1D k-means clustering on this PC for all possible
        # numbers of clusters
        k_clusters = []
        bics = []
        for j in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=j)
            k_clusters.append(kmeans.fit(col_reshaped))
            bics.append(compute_bic(kmeans, col_reshaped))

        # Choose the partition that results in the highest BIC
        best_k = k_clusters[np.argmax(bics)]
        k_results = [
            [x for x in np.where(best_k.labels_ == y)[0]] 
            for y in range(best_k.n_clusters)
        ]

        # Add each discovered cluster to the list of discovered clusters
        for cluster in k_results:
            cur_class = [sounds[idx] for idx in cluster]
            classes_list.append(cur_class)

        # Perform recursive clustering on all discovered classes we haven't
        # seen yet.
        for c in classes_list:
            # Check that we haven't already clustered this subet. This isn't
            # strictly necessary, but saves some cycles.
            if not c in visited_classes and len(c) > 1:
                visited_classes.append(c)

                # Perform recursive clustering on this subset.
                subidx = sorted([sounds.index(s) for s in c])
                subspace = input_data[subidx]
                subsounds = [s for s in sounds if s in c]
                found_subclasses = find_classes(
                    subspace, subsounds, v_scalar=v_scalar,
                    visited_classes=visited_classes
                )
                sub_classes.extend(found_subclasses)
        # Add classes from this call and all recursive calls to the list of
        # discovered classes.
        full_classes_list += classes_list + sub_classes

    # Returns founds classes
    return full_classes_list
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Performs a combination of PCA and 1D k-means clustering "
                      "to find phonological classes from an embedding."
    )
    parser.add_argument(
        'input_file_stem', type=str, help='The stem of the set of input files.'
    )
    parser.add_argument(
        'output_file', type=str, 
        help='Path to where the discovered classes will be saved.'
    )
    parser.add_argument(
        '--v_scalar', type=float, 
        help='A parameter that controls what amount of variance a principal '
        'component must account for to be used in clustering. The threshold '
        'is this value * (average amount of variance).',
        default=DEFAULT_VARIABILITY_SCALAR
    )
    parser.add_argument(
        '--no_constrain_initial_partition', action='store_false',
        help='A parameter that, if TRUE, sets restrictions on the initial '
        'partition of the data set: namely, any partition of the full set of '
        'sounds must be into two classes (e.g. consonants vs. vowels, voiced vs. '
        'voiceless, etc.)',
        default=DEFAULT_CONSTRAIN_PARTITIONS
    )
    parser.add_argument(
        '--no_constrain_initial_pcs', action='store_false', 
        help='A parameter that, if TRUE, restricts the initial partition of the '
        'data set: namely, only the first principal component is considered. '
        'Setting this to FALSE will result in the same classes being detected as '
        'when it is TRUE, but with additional partitions of the data set potentially '
        'discovered as well. Similar results can be gained by increasing '
        'the variability scalar, but this will apply to all recursive calls to the '
        'clusterer rather than just the top level call.',
        default=DEFAULT_CONSTRAIN_PCS
    )

    args = parser.parse_args()
    do_clustering(
        args.input_file_stem, args.output_file, args.v_scalar, 
        args.no_constrain_initial_partition, args.no_constrain_initial_pcs
    )
