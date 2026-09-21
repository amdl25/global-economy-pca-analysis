'''
Clasa care incapsuleaza implementarea modelului de ACP
'''
import numpy as np


class ACP:
    # asumam ca primit in constructor X ca numpy.ndarray
    def __init__(self, X):
        self.X = X
        # calcul matrice varianta-covarianta pentru X
        self.Cov = np.cov(m=X, rowvar=False)  # avem variabilele pe coloane
        print(self.Cov.shape)
        # calcul valorii proprii si vectori proprii pentru matricea varianta-covarianta
        self.valProp, self.vectProp = np.linalg.eigh(a=self.Cov)
        print(self.valProp, self.valProp.shape)
        print(self.vectProp.shape)
        k_desc = [k for k in reversed(np.argsort(a=self.valProp))]
        print(k_desc)
        self.alpha = self.valProp[k_desc]
        self.A = self.vectProp[:, k_desc]
        # regularizare vectori proprii
        for j in range(self.A.shape[1]):
            minCol = np.min(a=self.A[:, j])
            maxCol = np.max(a=self.A[:, j])
            if np.abs(minCol) > np.abs(maxCol):
                self.A[:, j] = (-1) * self.A[:, j]  # inmultirea cu un scalar nu afecteaza natura vectorului propriu

        # calcul componente principale
        # self.C = np.matmul(self.X, self.A)
        self.C = self.X @ self.A  # varianta cu supraincarcare de operator

        # calcul factori de corelatie (factor loadings)
        self.Rxc = self.A * np.sqrt(self.alpha)

        self.C2 = self.C * self.C
        # self.C2 = np.square(self.C)  # alternativa

    def getValProp(self):
        # return self.valProp
        return self.alpha
    def getVectProp(self):
        return self.A
    def getCompPrin(self):
        return self.C
    def getFactorLoadings(self):
        return self.Rxc



