from __future__ import division
import numpy as np
import scipy
import scipy.io
from sklearn.svm import SVC
import sklearn
from random import shuffle

class HMM():
    def __init__(self, initial, transitions, train_X, train_y, num_classes):
        """
        initial: length n python list of intial distribution for n states
        transitions: an n by n matrix of transition probabilities, ie:
        [[1/3, 2/3],
        [0, 1]]
        train_x: list of data to train the SVM on
        train_y: list of classes for training data (train_y[i] = label for train_x[i])
        (classes must be numbered, zero-indexed)
        num_classes: the number of classes in data
        """

        self.belief = np.array(initial)
        self.transitions = np.array(transitions)

        all_data = list(zip(train_X, train_y))
        shuffle(all_data)
        train_X = [i[0] for i in all_data]
        train_y = [i[1] for i in all_data]

        self.svm = SVC(kernel='linear', C=1, probability=True)
        self.svm.fit(train_X, train_y)
        self.num_classes = num_classes
        self.num_updates = 0
        self.num_wrong = 0

    def update(self, obs, c):
        # obs = obs[:3]
        self.num_updates += 1
        scores = self.svm.predict_proba(obs)[0]
        t = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            t[i] = np.sum(self.transitions[i]) * self.belief[i]
        self.belief = self.kludge(np.multiply(scores, t))
        self.belief /= np.sum(self.belief)
        if(np.argmax(self.belief) != c):
            self.num_wrong += 1
        return self.belief

    def update_live(self, obs):
        # obs = obs[:3]
        scores = self.svm.predict_proba(obs)[0]
        t = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            t[i] = np.sum(self.transitions[i]) * self.belief[i]
        self.belief = self.kludge(np.multiply(scores, t))
        self.belief /= np.sum(self.belief)
        return self.belief

    def accuracy(self):
        return 1 - (self.num_wrong / self.num_updates)

    def kludge(self, b):
        new = b + np.array([1e-4, 1e-4, 1e-4])
        return new / sum(new)
