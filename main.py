from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from utils import extract_100_images
from KMeans import KMeans

def find_optimal_k(train_x, train_y, method):
    j_clusts = []
    for k in range(5, 20):
        k_means = KMeans(train_x, train_y, k, method)
        k_means.train(100)
        j_clusts.append(k_means.j[-1])
    j_clusts = np.array(j_clusts)
    return j_clusts

def plot_J(title, x, y):
    fig = plt.figure()
    fig.suptitle(title, fontsize=20)
    plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    train, test = extract_100_images()
    train_x = train[:, :-1]
    train_y = train[:, -1]

    test_x = test[:, :-1]
    test_y = test[:, -1]
    
    # Random Initialization
    k_means = KMeans(train_x, train_y, 20, 'random')
    # train for 100 iterations
    iters = k_means.train(100)
    print(f'The training convered at {iters}')

    # Plot the cluster reps
    # Part a :
    k_means.plot()
    print(f'Final Cluster Loss : {k_means.j[-1]}')

    # Part b : 
    accuracy = k_means.accuracy(test_x, test_y)
    print(f'The Accuracy on the 50 images is : {accuracy}')

    j_clusts = find_optimal_k(train_x, train_y, 'random')
    min_j_val = np.min(j_clusts)
    min_j_k = np.argmin(j_clusts)+5
    print(f'The Mininum J value occurs at k={min_j_k} and is {min_j_val}')
    plot_J("J_Cluster vs k", range(5, 20), j_clusts)
###############
    ##### Datapoint initialisation
    k_means = KMeans(train_x, train_y, 20, 'dataset')
    # train for 100 iterations
    iters = k_means.train(100)
    print(f'The training convered at {iters}')

    # Plot the cluster reps
    # Part a :
    k_means.plot()
    print(f'Final Cluster Loss : {k_means.j[-1]}')

    # Part b : 
    accuracy = k_means.accuracy(test_x, test_y)
    print(f'The Accuracy on the 50 images is : {accuracy}')

    j_clusts = find_optimal_k(train_x, train_y, 'dataset')
    min_j_val = np.min(j_clusts)
    min_j_k = np.argmin(j_clusts)+5
    print(f'The Mininum J value occurs at k={min_j_k} and is {min_j_val}')
    plot_J("J_Cluster vs k", range(5, 20), j_clusts)