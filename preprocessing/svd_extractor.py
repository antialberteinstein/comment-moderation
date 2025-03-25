import numpy as np

class SVDExtractor:

    def __init__(self, reverse=False, k=None, rate=None):
        # reverse = True: documents are the rows.
        # reverse = False: documents are the columns.
        # k = number of used features.
        # rate = (number of used features)/(number of all features).
        # unused features will be eliminated.
        self._reverse = reverse
        self._features_matrix = None
        self._rate = None
        self._k = None
        if rate is not None:
            self._rate = rate
        elif k is not None:
            self._k = k

    @property
    def features_matrix(self):
        return self._features_matrix

    def fit_transform(self, tcw_matrix):
        self._fit(tcw_matrix)

    def _fit(self, tcw_matrix):
        if self._reverse:
            U, _, _ = np.linalg.svd(tcw_matrix)
        else:
            U, _, _ = np.linalg.svd(tcw_matrix.T)

        # Number of used features.
        _n = U.shape[1]
        if self._k != None:
            if self._k <= _n:
                _n = self._k
        elif self._rate != None:
            _n = int(_n * self._rate)

        self._features_matrix = U[:, :_n]

