
class KNeighborsClassifier:
    """c : percentage of data to be used per label"""
    def __init__(self, data, labels, c=None):
        self.data = data
        self.labels = labels
        self.c = c
        self.cut_data = {}

    def fit(self):
        if not self.c:
            from KNeighborsARC.KNeighborsARC import KNeighborsClassifier
            self.Classifier = KNeighborsClassifier(data=self.data, labels=self.labels)

        if self.c:
            print('# Limiting fit ...')
            self.limit_fit()
            from KNeighborsARC.KNearestARC import KNearestClassifier
            print(self.cut_data)
            self.Classifier = KNearestClassifier(self.cut_data)

    def predict(self, array, k=3, show_step=1):
        return self.Classifier.predict(array, k, show_step)

    def limit_fit(self):
        _labels = []
        [_labels.append(_) for _ in self.labels if _ not in _labels]
        for group in _labels:
            self.cut_data[group] = []
        for _ in range(len(self.data)):
            print(self.labels[_])
            if len(self.cut_data[self.labels[_]]) < self.c:
                self.cut_data[self.labels[_]].append(self.data[_])
            if all([len(__) == self.c for _group in _labels for __ in self.cut_data[_group]]):
                break




