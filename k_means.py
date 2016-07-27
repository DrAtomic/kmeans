# Tyler Sorg
# Machine Learning
# K-Means Clustering using Optdigit dataset
import itertools
import numpy as np
from PIL import Image

number_of_clusters = 10
number_of_classes = 10  # digits 0 to 9
number_of_trials = 1


def load_data(filename="optdigits.train"):
    unprocessed_data_file = file('./optdigits/' + filename, 'r')
    unprocessed_data = unprocessed_data_file.readlines()

    labels = []
    features = []

    for line in unprocessed_data:
        feature_vector = []
        split_line = line.split(',')
        for element in split_line[:-1]:
            feature_vector.append(float(element))
        features.append(feature_vector)
        labels.append(int(split_line[-1]))

    return features, labels


def distance(point, center):
    """
    Parameters
    ----------
    point: an n_dimensional data point
    center: an n_dimensional center

    Returns
    -------
    A floating point value representing the distance between a given point
    and center pair.
    """
    square_sums = 0.0
    for point_i, center_i in zip(point, center):
        square_sums += (point_i - center_i) ** 2
    return np.sqrt(square_sums)


def closest_center(point, centers):
    """

    Parameters
    ----------
    point: a single data point
    centers: the list of center points

    Returns
    -------
    Returns the index of the closest center to the point. If there are more
    than one closest points, one is randomly selected.

    """
    distances = list()
    for center in centers:
        distances.append(distance(point, center))
    dist_array = np.array(distances)

    first_min_distance = dist_array.argmin()

    min_distances = list()
    for i in range(len(distances)):
        if distances[i] - distances[first_min_distance] < 10 ** -10:
            min_distances.append(i)
    return np.random.choice(min_distances)


def random_center():
    """

    Returns
    -------
    Returns a 64 element tuple of random integers between 0-16

    """
    return np.random.randint(0, 16, 64).tolist()


def sum_squared_error(clustering, centers, data):
    error = 0
    for i in range(number_of_clusters):
        cluster = clustering[i]
        center = centers[i]
        for data_point_index in cluster:
            datapoint = data[data_point_index]
            error += distance(datapoint, center) ** 2
    return error


def sum_squared_separation(clustering, centers):
    # check if clusters i, j are both nonempty first.
    # generate pairs where order doesn't matter... num_clusters choose 2
    pairs = itertools.combinations([i for i in range(number_of_clusters)], 2)
    separation = 0
    for pair in pairs:
        separation += distance(centers[pair[0]], centers[pair[1]]) ** 2
    return separation


def entropy(cluster, labels):
    entropy_sum = 0

    class_representation_in_cluster = [0 for i in range(number_of_classes)]
    total_instances = len(cluster)

    if total_instances == 0:  # Special case: 0*log_2(0) is just 0, okay?
        return 0

    for point in cluster:  # TODO: Check if this counts correctly.
        # point is an index stored in the cluster.
        # data[point] is the datapoint at that index,
        # labels[point] is the corresponding class..
        # count each by incrementing index 0, 1, ..., or 9 for every point in
        # cluster.
        class_representation_in_cluster[labels[point]] += 1

    class_ratios = [float(class_representation_in_cluster[i]) / total_instances
                    for i in range(number_of_classes)]
    for i in range(number_of_classes):
        if class_representation_in_cluster[i] < 1:  # Let Log_2(0) = 0
            product = 0.0
        else:
            product = class_ratios[i] * np.log2(class_ratios[i])
        entropy_sum += product

    return -1 * entropy_sum


def mean_entropy(clustering, labels):
    """

    Parameters
    ----------
    clustering = the list of clusters

    Returns
    -------
    Return mean entropy of the clusters in the clustering.

    Notes
    -----
    Want to minimize mean entropy.

    """
    instances_per_cluster = [len(cluster) for cluster in clustering]

    total_number_of_instances = sum(instances_per_cluster)

    ratios = [float(instances_per_cluster[i]) / total_number_of_instances \
              for i in range(number_of_clusters)]

    weighted_entropies = [ratios[i] * entropy(clustering[i], labels) \
                          for i in range(number_of_clusters)]

    mean = float(sum(weighted_entropies)) / len(weighted_entropies)

    return mean


def set_number_of_clusters(number):
    global number_of_clusters
    number_of_clusters = number


def check_if_centers_updated(old_centers, centers):
    difference = 0
    for old, new in zip(old_centers, centers):
        difference += np.sum(np.abs(np.array(old) - np.array(new)))
    if difference < 10 ** -1:  # if the difference is arbitrarily close to 0
        return True
    else:
        return False


def most_popular_class(cluster, labels):
    class_representation_in_cluster = [0 for i in range(number_of_classes)]
    total_instances = len(cluster)

    if total_instances == 0:  # Special case: 0*log_2(0) is just 0, okay?
        return None

    for point in cluster:  # TODO: Check if this counts correctly.
        # point is an index stored in the cluster.
        # data[point] is the datapoint at that index,
        # labels[point] is the corresponding class..
        # count each by incrementing index 0, 1, ..., or 9 for every point in
        # cluster.
        class_representation_in_cluster[labels[point]] += 1

    most_popular_count = max(class_representation_in_cluster)
    first_most_popular_index = class_representation_in_cluster.index(
        most_popular_count)

    if class_representation_in_cluster.count(most_popular_count) is 1:
        return first_most_popular_index
    else:  # The cluster has equal representation of two or more classes.
        indices_of_tied_classes = []
        for c in class_representation_in_cluster:
            if c == most_popular_count:
                indices_of_tied_classes.append(
                    class_representation_in_cluster.index(c))

        return np.random.choice(indices_of_tied_classes)


def classify(centers, cluster_to_class, test):
    closest = closest_center(test, centers)
    return cluster_to_class[closest]


def k_means(testing_features, testing_labels, training_features,
            training_labels):
    k_means_trials = dict()  # save data about each trial
    for trial in range(number_of_trials):
        print 'Beginning trial #%d...' % (trial)
        # 2: Initialize centers for each cluster randomly
        print 'Initializing random centers for %d clusters..' % (
            number_of_clusters)
        centers = [random_center() for i in range(number_of_clusters)]

        # Repeat 3-5 until centers don't move or they oscillate
        print 'Working on finding cluster centers..'
        no_change = False

        # Stop iterating K-Means when all cluster centers stop changing or if
        # the algorithm is stuck in an oscillation.
        while no_change is False:
            # 3: Calculate closest centers for each data point
            closest_centers = []
            for datapoint in training_features:
                closest_centers.append(closest_center(datapoint, centers))
            clustering = [[] for i in range(number_of_clusters)]
            for i in range(len(closest_centers)):
                clustering[closest_centers[i]].append(i)

            # 4: Calculate the centroid of each center's set of points and
            centroids = []
            for cluster in clustering:
                mean_vector = np.array([0.0 for i in range(64)])  # sum feature
                # values
                for i in range(len(cluster)):  # for each point in cluster
                    # sum the features
                    mean_vector += np.array((training_features[cluster[i]]))
                if len(cluster) > 0:
                    mean_vector /= float(len(cluster))  # average the sums
                centroids.append(mean_vector)

            # 5: Reassign each center to the centroid's location.
            old_centers = centers
            centers = centroids
            no_change = check_if_centers_updated(old_centers, centers)
            print '.',
        print '\nFinal cluster centers for trial %d set.' % (trial)
        sse = sum_squared_error(clustering, centers, training_features)
        sss = sum_squared_separation(clustering, centers)
        avg_entropy = mean_entropy(clustering, training_labels)

        k_means_trials[trial] = [centers, closest_centers, clustering, sse,
                                 sss, avg_entropy]

    # Choose the run (out of 5) that yields the smallest sum-squared error
    # - For this best run, in your report give the sum-squared error,
    # - sum-squared separation, and mean entropy of the resulting clustering.
    print 'All %d trials are completed.\n' % (number_of_trials)
    for i in range(number_of_trials):
        print '\nTrial#', str(i), ':'
        print 'SSE: ', k_means_trials[i][3]
        print 'SSS: ', k_means_trials[i][4]
        print 'Mean Entropy: ', k_means_trials[i][5]
    smallest_sse_index = 0
    for trial in range(1, len(k_means_trials)):
        if k_means_trials[trial][3] < k_means_trials[smallest_sse_index][3]:
            smallest_sse_index = trial
    print '\nThe best trial was number %d' % smallest_sse_index
    best_trial = k_means_trials[smallest_sse_index]
    best_centers = best_trial[0]
    best_clustering = best_trial[2]

    # - Now use this clustering to classify the test data, as follows:
    # - Associate each cluster center with the most frequent class it
    # - contains. If there is a tie for most frequent class, break the tie
    # - at random.
    print 'Assigning classes to each cluster based on popularity.'
    cluster_labels = [most_popular_class(cluster, training_labels) for
                      cluster in best_clustering]
    # - Assign each test instance the class of the closest cluster center.
    # - Again, ties are broken at random. Give the accuracy on the test data
    # - as well a confusion matrix.
    print 'Assigning classifications to each test instance.'
    classifications = [classify(best_centers, cluster_labels, test) for
                       test in testing_features]
    # - Calculate the accuracy on the test data and create a confusion matrix
    # - for the results on the test data.
    confusion_matrix = create_confusion_matrix(classifications, testing_labels)
    save_confusion_matrix(confusion_matrix)

    return [best_trial, cluster_labels, classifications, confusion_matrix]


def create_confusion_matrix(classifications, testing_labels):
    confusion_matrix = [[0 for i in range(number_of_classes)] for i in
                        range(number_of_classes)]
    for label, classification in zip(testing_labels, classifications):
        confusion_matrix[label][classification] += 1
    return confusion_matrix


def save_confusion_matrix(confusion_matrix):
    filename = 'confusion_matrix_%d_clusters.csv' % (number_of_clusters)
    output = open(filename, 'w')
    for row in confusion_matrix:
        for col in row:
            output.write(str(col) + ',')
        output.write('\n')
    output.close()


def display_confusion_matrix(confusion_matrix):
    print '\nConfusion matrix for K=%d:' % (number_of_clusters)
    for row in confusion_matrix:
        for col in row:
            print col, ',',
        print '\n'


def accuracy(confusion_matrix):
    m = np.array(confusion_matrix)
    return float(np.sum(np.diagonal(m))) / np.sum(m)


def pixel_value(value):
    value = int(np.floor(value))
    return value * 16


def draw_center_as_bitmap(name_prefix, center_number, center):
    img = Image.new('L', (8, 8), "black")
    center_2d = np.array(center).reshape(8, 8)
    for i in range(img.size[0]):
        for j in range(img.size[0]):
            # img.putpixel((i, j), pixel_value(int(center_2d[i][j]))) # weird
            #  rotation going on...
            img.putpixel((j, i), pixel_value(int(center_2d[i][j])))
            # img.putpixel((i, j), pixel_value(int(center_2d[j][i])))
    name = name_prefix + str(center_number) + '.png'
    img.save(name)


def record_cluster_labels(filename, cluster_labels):
    cluster_labels_recorder = open(filename, 'w')
    cluster_labels_recorder.write('Labels of each cluster:\n')
    for i in range(len(cluster_labels)):
        if cluster_labels[i] is not None:
            cluster_labels_recorder.write('Cluster %d\'s label is %d\n' %
                                          (i, cluster_labels[i]))
        elif cluster_labels[i] is None:
            cluster_labels_recorder.write('Cluster %d\'s label is None\n' %
                                          (i))


def main():
    training_features, training_labels = load_data()
    testing_features, testing_labels = load_data('optdigits.test')

    # ==========================================================================
    # Experiment 1: Repeat the following 5 times, with different
    # random number seeds.
    # ==========================================================================
    set_number_of_clusters(10)
    ten_clusters_results = k_means(testing_features, testing_labels,
                                   training_features,
                                   training_labels)
    accuracy_10_clusters = accuracy(ten_clusters_results[-1])
    print 'Accuracy of 10 clusters: ', accuracy_10_clusters
    ten_centers = ten_clusters_results[0][0]
    for i in range(number_of_clusters):
        draw_center_as_bitmap('exp1_center_', i, ten_centers[i])

    ten_cluster_labels = ten_clusters_results[1]
    record_cluster_labels('exp1_cluster_labels.txt', ten_cluster_labels)

    # ==========================================================================
    # Experiment 2: Run K-means on the same data but with K = 30. Calculate
    # the same things for comparison.
    # ==========================================================================
    set_number_of_clusters(30)
    thirty_clusters_results = k_means(testing_features, testing_labels,
                                      training_features,
                                      training_labels)
    accuracy_30_clusters = accuracy(thirty_clusters_results[-1])
    print 'Accuracy of 30 clusters: ', accuracy_30_clusters
    thirty_centers = thirty_clusters_results[0][0]
    for i in range(number_of_clusters):
        draw_center_as_bitmap('exp2_center_', i, thirty_centers[i])
    thirty_cluster_labels = thirty_clusters_results[1]
    record_cluster_labels('exp2_cluster_labels.txt', thirty_cluster_labels)


if __name__ == "__main__":
    main()
