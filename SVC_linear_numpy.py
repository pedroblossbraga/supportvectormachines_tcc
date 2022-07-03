"""
Código modificado a partir das referências 
- https://towardsdatascience.com/implementing-svm-from-scratch-784e4ad0bc6a
- https://fordcombs.medium.com/svm-from-scratch-step-by-step-in-python-f1e2d5b9c5be
- https://github.com/marvinlanhenke/DataScience/blob/main/MachineLearningFromScratch/SVM/train.py
por Pedro Blöss Braga
"""
import os
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.rcParams['axes.facecolor']='white'

class SVC_linear:
    def __init__(self, 
                 learning_rate=1e-3, 
                 lambda_param=1e-2, 
                 n_iters=1000):
        self.eta = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def _init_weights_bias(self, X):
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0

    def _get_cls_map(self, y):
        # \mathbb{I}(y<=0)
        return np.where(y <= 0, -1, 1)

    def _satisfy_constraint(self, x, idx):
        # y_i (<w, x_i> + b ) >= 1
        linear_model = np.dot(x, self.w) + self.b 
        return self.cls_map[idx] * linear_model >= 1
    
    def _get_gradients(self, constrain, x, idx):
        if constrain:
            dw = self.lambda_param * self.w
            db = 0
            return dw, db
        # dw <- lambda w - <y, x>
        dw = self.lambda_param * self.w - np.dot(self.cls_map[idx], x)
        db = - self.cls_map[idx]
        return dw, db
    
    def _update_weights_bias(self, dw, db):
        # w_{t+1} <-  w - eta (lambda w_t -  <y, x>) =
        #             w - (eta lambda w_t - eta <y, x>) =
        #            (1 - eta lambda )wt + eta <y,x>
        self.w -= self.eta * dw
        self.b -= self.eta * db
    
    def fit(self, X, y):

        # set w_1 = 0
        self._init_weights_bias(X)
        self.cls_map = self._get_cls_map(y)

        for _ in range(self.n_iters): # for t = 1, .., T

            for idx, x in enumerate(X):

                constrain = self._satisfy_constraint(x, idx)
                dw, db = self._get_gradients(constrain, x, idx)
                self._update_weights_bias(dw, db)
    
    def predict(self, X):
        estimate = np.dot(X, self.w) + self.b
        prediction = np.sign(estimate)
        return np.where(prediction == -1, 0, 1)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true==y_pred) / len(y_true)
    return accuracy

def get_hyperplane(x, w, b, offset):
    # <w,x> + b 
    return (-w[0] * x - b + offset) / w[1]

def main():
    # criando dados fictícios
    X, y = datasets.make_blobs(
        n_samples=250, 
        n_features=2, 
        centers=2, 
        # cluster_std=1.05,
        cluster_std = 3,
         random_state=1
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
    test_size=0.1,
     shuffle=True, 
    random_state=42)

    # instanciando o classificador
    clf = SVC_linear(n_iters=1000)

    # treinando o classificador
    clf.fit(X_train, y_train)

    # efetuando predições
    predictions = clf.predict(X_test)

    print("SVC Accuracy: ", accuracy(y_test, predictions))

    # plotando os resultados
    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = get_hyperplane(x0_1, clf.w, clf.b, 0)
    x1_2 = get_hyperplane(x0_2, clf.w, clf.b, 0)

    x1_1_m = get_hyperplane(x0_1, clf.w, clf.b, -1)
    x1_2_m = get_hyperplane(x0_2, clf.w, clf.b, -1)

    x1_1_p = get_hyperplane(x0_1, clf.w, clf.b, 1)
    x1_2_p = get_hyperplane(x0_2, clf.w, clf.b, 1)


    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])

    fig, ax = plt.subplots(1, 1, figsize=(10,6))

    # plt.set_cmap('PiYG')
    plt.scatter(X_train[:, 0], X_train[:, 1], 
                marker='o', 
                c=y_train, 
                s=100, 
                alpha=0.75)
    plt.scatter(X_test[:, 0], X_test[:, 1], 
                marker="x", 
                c=y_test, 
                s=100, 
                alpha=0.75)
    plt.fill_between(
        [-13, 1],
        [x1_1_m, x1_2_m],
        [-x0_2, x1_2_p], 
        color='gold',
        interpolate=False,
        alpha=0.5,
        label='margem'
    )
    plt.plot(
        [x0_1, x0_2], [x1_1, x1_2],
        linestyle='-',
        # color='k',
        color='blue',
        lw=1,
        alpha=0.9,
        label='hiperplano de decisão'
    )
    plt.plot(
        [x0_1, x0_2], [x1_1_m, x1_2_m], 
        linestyle="--", 
        # color='grey', 
        color='red',
        lw=1, 
        alpha=0.8,
        # label='upper support hyperplane'
        label='hiperplanos de suporte'
    )
    plt.plot(
        [x0_1, x0_2], [x1_1_p, x1_2_p], 
        linestyle="--", 
        # color='grey', 
        color='red',
        lw=1, 
        alpha=0.8,
        # label='lower support hyperplane'
    )
    ax.set_ylim([x1_min - 3, x1_max + 3])
    for spine in ['top','right']:
        ax.spines[spine].set_visible(False)
    plt.legend(
            bbox_to_anchor = (1.05, 0.9),
            title='Hiperplanos',
            title_fontsize=16,
            fontsize=14
            )
    plt.text(-3, -4.5,
            r'$\langle w, x \rangle + b = - 1$',
            color='red',
            fontsize = 15)
    plt.text(-12.5, 4.5,
            r'$\langle w, x \rangle + b = 1$',
            color='red',
            fontsize = 15)
    plt.text(-10, 1.1,
            r'$\langle w, x \rangle + b = 0$',
            color='blue',
            fontsize = 15)
    plt.text(-10, -8, 
            r'$\{y:\mathcal{C}(y)=-1\}$',
            color='green',
            fontsize = 15)
    plt.text(-3, 7.8, 
            r'$\{y:\mathcal{C}(y)=+1\}$',
            color='darkmagenta',
            fontsize = 15)
    plt.title('Classificação com vetores de suporte e hiperplano ótimo',
            fontsize = 16)
    # plt.show()
    plt.savefig(
        os.path.join(os.getcwd(), 'images', 'SVC_blob_margin.png')
    )

if __name__ == "__main__":
    main()