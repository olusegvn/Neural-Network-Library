import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
from queue import Queue
from threading import Thread as thread
np.set_printoptions(threshold=np.inf)
style.use('fivethirtyeight')


class KNearestClassifier:
    def __init__(self, data):
        self.data = data
        self.vote = ''
        self.pred = []
        self.results = []

    def predict(self, predict_array, k=3, show_step=1):
        self.results = []
        counter = 1
        for _list in predict_array:
            self.pred = _list
            distance = []
            start_time = time.time()
            f = time.time()
            group = 0
            feature = self.data[group]
            for i in range(1):
                for j in range(1):
                    euclidean_distance = np.linalg.norm(np.array(feature) - np.array(_list))
                    distance.append([euclidean_distance, group])
            votes = [i[1] for i in sorted(distance)[:k]]
            # print("Top votes : ", votes)
            vote = Counter(votes).most_common(1)[0][0]
            # print("Vote: ", vote)
            self.vote = vote
            self.results.append(vote)
            _time = time.time() - start_time
            if counter % show_step == 0:
                print(round((counter/len(predict_array)) * 100, 2), '%  :\t', counter, 'of', len(predict_array),
                      'classified |\t ETA : ', datetime.timedelta(seconds=_time * (len(predict_array) - counter)))
            counter += 1

        return self.results

    def show(self):
        for i in self.data:
            for j in i[1]:
                plt.scatter(j[0], j[1], color=i)
                plt.scatter(self.pred[0], self.pred[1], color=self.vote)
        plt.show()

    def check_accuracy(self, labels):
        error = (np.array(self.results) - np.array(labels)).sum()
        accuracy = 100 - (error * 100)
        return accuracy, '%'

