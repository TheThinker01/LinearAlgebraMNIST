import numpy as np
import matplotlib.pyplot as plt
class KMeans():
    def __init__(self, x_train, y_train, cluster_ct, init_method):
        self.j = []
        self.data_size = x_train.shape[0]
        self.num_features = x_train.shape[1]
        self.x_train = x_train
        self.y_train = y_train
        self.clusters_count = cluster_ct
        self.labels = np.zeros(self.data_size)
        if init_method == 'dataset':
            self.centroids = self.x_train[np.random.randint(0, self.data_size, self.clusters_count)]
        else :
            self.centroids = np.random.uniform(size=(cluster_ct, self.num_features))

    def train(self, num_iters):
        iter = None
        for iters in range(num_iters):
            print(f'At Iteration {iters} ...')
            # step 1 
            self.update_clusters()
            # step 2
            self.update_centroids()
            self.j.append(self.loss())
            if self.converged():
                iter = iters
                break

        return iter

    def update_clusters(self):
        for i in range(self.data_size):
            self.labels[i] = int(np.argmin(np.array([self.distance(self.x_train[i, :], x) for x in self.centroids])))

    def distance(self, x, y):
        return np.linalg.norm(x-y, 2)

    def update_centroids(self):
        for i in range(self.clusters_count):
            index = np.where(self.labels == i)[0]
            if index.shape[0] > 0 :
                self.centroids[i, :] = np.mean(self.x_train[index], axis=0)
            else:
                self.centroids[i,:] = np.zeros(self.num_features)
    
    def loss(self):
        loss = np.array([self.distance(self.x_train[i, :],
                                                 self.centroids[int(self.labels[i]), :]) for i in range(self.data_size)])
        return np.mean(loss)

    def converged(self):
        if len(self.j) < 2:
            return False
        if self.j[-1] == self.j[-2]:
            return True
        return False
    
    def plot(self):
        image_size = int(np.sqrt(self.num_features))
        Z = self.centroids.reshape((self.clusters_count, image_size, image_size))
        fig, axs = plt.subplots(5, 4, figsize=(14, 14))
        plt.gray()
        for i, ax in enumerate(axs.flat):
            ax.matshow(Z[i])
            ax.axis('off')
        fig.suptitle("Cluster Representatives", fontsize=30)
        plt.show()

    def update_clusters2(self, data, labels):
        for i in range(data.shape[0]):
            labels[i] = int(np.argmin(
                np.array([self.distance(data[i, :], x) for x in self.centroids])))
        return labels

    def accuracy(self, test_x, test_y):
        self.cluster_target = -1*np.ones(self.clusters_count)
        for i in range(self.clusters_count):
            temp = np.bincount(np.array(self.y_train[np.where(self.labels == i)],dtype=np.int))
            if temp.shape[0] > 0:
                self.cluster_target[i] = int(np.argmax(temp, axis=0))

        cluster_labels = np.zeros(test_x.shape[0], dtype=np.int)
        cluster_labels = self.update_clusters2(test_x, cluster_labels)
        predictions = self.cluster_target[cluster_labels]
        accuracy = np.mean(
            np.array(np.equal(predictions, test_y), dtype=np.float32))
        return accuracy
