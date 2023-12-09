import pandas as pd
import seaborn as sns                   
import matplotlib.pyplot as plt   

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix


def perform_regression_analysis(df):
    df['neo'].replace({'Y': 1, 'N': 0}, inplace=True)

    print("\nКоличество ближных астероидов:")
    print(df['neo'].value_counts())

    # Определение признаков и целевой переменной
    X = df[["diameter", "diameter_sigma", "a", "i", "ad",  "moid_ld", "moid",  "moid_ld"]]
    Y = df["neo"]

    # Создание и обучение модели логистической регрессии
    model = LogisticRegression(max_iter=1000)
    model.fit(X, Y)

    # Получение статуса и точности модели
    status = model.predict(X)
    accuracy = accuracy_score(Y, status)
    slope = model.coef_

    # Вывод коэффициентов и точности модели
    print("Coefficients:", slope)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Вывод confusion matrix
    conf_matrix = confusion_matrix(Y, status)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Отображение confusion matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Не близко', 'Близко'], yticklabels=['Не близко', 'Близко'])
    plt.xlabel('Предсказано')
    plt.ylabel('Фактически')
    plt.title('Confusion Matrix')
    plt.show()
