import numpy as np
from sklearn.cluster import KMeans
from contextlib import suppress
class DissimilarVectorsKMeans(KMeans):
    '''
    This class helps in clustering dissimilar vectors of varying length
    '''

    def __init__(self, n_clusters, init='k-means++', n_init=10, max_iter=300, tol=1e-4,
                 verbose=0, random_state=None, copy_x=True, algorithm='lloyd'):
        super().__init__(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                         verbose=verbose, random_state=random_state, copy_x=copy_x, algorithm=algorithm)
        
        
    def build_feature_table(self, X):
        '''
        Builds a list of all unique features in all the input dict vectors
        :param X: input dict vectors
        :return: List of features
        '''
        features = set()
        for datapoint in X:
            for feature in datapoint.keys():
                features.add(feature)
        return list(features)
        
    def normalize(self, x):
        '''

        Normalizes dissimilar vectors using feature table and saves all values in enlongated vectors
        :param x: dict of features => values
        :return: np array of values whose indices correspond to features in feature table
        '''
        #print(f"normalize X : {x}")
        if not hasattr(self, 'features'):
            raise AttributeError('feature table not built')
        total_features = len(self.features)
        #print(f"total_features : {total_features}")
        vector = np.zeros(shape=(total_features,))
        for attr, value in x.items():
            with suppress(ValueError):
                index = self.features.index(attr)
                vector[index] = value
        #print(f"vector : {vector}")
        return vector

    def build_vectors(self, X):
        '''
        Normalizes all input dict vectors
        :param X: input dict vectors
        :return: np array of normalized vectors
        '''
        #print(f"build_vectors X : {X}")
        vectors = []
        for index in range(len(X)):
            vector = self.normalize(X[index])
            vectors.append(vector)
        #print(f"vectors : {vectors}")
        return np.array(vectors)

    def denormalize(self, x):
        '''
        Builds dictionary based on feature table and non zero valued indices
        :param x: normalized np array
        :return: dict of features => values
        '''
        if not hasattr(self, 'features'):
            raise AttributeError('feature table not built')
        vector = {}
        for i in range(len(x)):
            if x[i] == 0:
                continue
            feature = self.features[i]
            vector[feature] = x[i]
        return vector

    def build_featured_centers(self):
        '''
        Denormalizes the exisiting cluster_centers_
        :return: list of denormalized cluster centers
        '''
        cluster_centers = []
        for cluster_center in self.cluster_centers_:
            denormalized_center = self.denormalize(cluster_center)
            cluster_centers.append(denormalized_center)
        return cluster_centers

    def fit(self, X, y=None):
        '''
        Trains the model
        :param X: input dict vectors
        :param y: None
        :return: labels
        '''
        self.features = self.build_feature_table(X)
        # print("feat: ", self.features)
        normalized_X = self.build_vectors(X)
        # print("normalized_X: ",normalized_X)
        labels = super().fit(normalized_X)
        self.cluster_centers = self.build_featured_centers()
        return labels

          
    def predict(self, X):
        '''
        Predicts the output based on trained model
        :param X: input dict vector
        :return: label of predicted center
        '''
        if not hasattr(self, 'features'):
            raise AttributeError('feature table not built')
        if not hasattr(self, 'cluster_centers'):
            raise ValueError('model not trained yet. Consider call predict() after fit()')
        normalized_X = self.build_vectors(X)
        return np.array(super().predict(normalized_X))