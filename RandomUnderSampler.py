"""
Sub-amostrador pseudo-aleatório para dados de classificação dicotômica
desbalanceada.
Autor: Pedro Blöss Braga
"""
from collections import Counter
import pandas as pd

class MyUnderSampler:
  def __init__(self,
               random_state = 42):
    self.random_state = random_state
  
  def fit_resample(self, X,y):
    self.X = X
    self.y = y

    # contando a frequência das classes $\{ | \{ y : \mathcal{C}(y)=c_k \} | \}_k$
    counter = Counter(y)

    # argmin_{i} {freq_i}
    min_class = list(dict(counter).values()).index(min(counter.values()))
    other_class = list(set(counter.keys()) - set([min_class]))[0]

    # menor frequência $q = \min (\{ | \{ y : \mathcal{C}(y)=c_k \} | \}_k)$
    min_freq = counter[min_class]

    # criando tabela com os dados
    df = pd.DataFrame(X)
    df['y'] = y

    # amostrando na frequência da menor classe
    df_other = df[df.y == other_class].sample(n = min_freq,
                                              random_state = self.random_state)
    df_min = df[df.y == min_class]

    # joining dataframes
    df_resampled = pd.concat([df_other, df_min], axis = 0)
    df_resampled = df_resampled.reset_index(drop=True)

    # to int
    df_resampled['y'] = df_resampled['y'].astype(int)

    # separando as features da variável target
    X_resampled = df_resampled.loc[:, df.columns != 'y'].values
    y_resampled = df_resampled['y'].values

    return X_resampled, y_resampled