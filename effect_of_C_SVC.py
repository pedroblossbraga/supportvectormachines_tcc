"""
Fonte: https://scikit-learn.org/stable/auto_examples/svm/plot_svm_margin.html
Modificado por Pedro Blöss
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import svm

# we create 40 separable points
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20

# figure number
fignum = 1

# margem regular, rígida e suave, respectivamente
classificadores = [("regular", 1, 'margem regular'), 
                   ("suave", 0.05, 'margem suave'), 
                   ('rigido', 100, 'margem rigida')]

# fazendo o fit dos modelos
for name, penalty, title in classificadores:

  # classificador com seu parâmetro de penalidade
  clf = svm.SVC(kernel="linear", C=penalty)
  clf.fit(X, Y) # treinamento

  # calculando o hiperplano de separação
  w = clf.coef_[0]
  a = -w[0] / w[1]
  xx = np.linspace(-5, 5)
  yy = a * xx - (clf.intercept_[0]) / w[1] # hiperplano de separação

  # calculando a margem
  margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))

  # paralelas ao hiperplano de separação
  yy_down = yy - np.sqrt(1 + a ** 2) * margin # hiperplano inferior
  yy_up = yy + np.sqrt(1 + a ** 2) * margin # hiperplano superior

  # ---- gráfico
  plt.figure(figsize=(8,2))

  # linhas dos hiperplanos
  plt.clf()
  plt.plot(xx, yy, "k-")
  plt.plot(xx, yy_down, "k--")
  plt.plot(xx, yy_up, "k--")

  # vetores mais próximos ao hiperplano de separação
  plt.scatter(
      clf.support_vectors_[:, 0],
      clf.support_vectors_[:, 1],
      s=80,
      facecolors="none",
      zorder=10,
      edgecolors="k",
      cmap=cm.get_cmap("RdBu"),
  )
  # pontos
  plt.scatter(
      X[:, 0], X[:, 1], 
      c=Y,
       zorder=10, 
      cmap=cm.get_cmap("RdBu"), 
      edgecolors="k"
  )

  plt.axis("tight")
  x_min = -4.8
  x_max = 4.2
  y_min = -6
  y_max = 6

  YY, XX = np.meshgrid(yy, xx)
  xy = np.vstack([XX.ravel(), YY.ravel()]).T
  Z = clf.decision_function(xy).reshape(XX.shape)

  # Put the result into a contour plot
  plt.contourf(XX, YY, Z, 
               cmap=cm.get_cmap("RdBu"), 
               alpha=0.5, 
               linestyles=["-"])

  plt.xlim(x_min, x_max)
  plt.ylim(y_min, y_max)

  plt.xticks(())
  plt.yticks(())
  plt.title('{}, C={}'.format(title, penalty), 
            fontsize=12)
  plt.show()