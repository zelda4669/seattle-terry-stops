import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix, classification_report, precision_score, recall_score, accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

def metrics(model, x_test, y_test):
    plot_confusion_matrix(model, x_test, y_test, cmap=plt.cm.Blues)
    plt.show()
    print(classification_report(y_test, model.predict(x_test)))
    print('\n')

def make_model(model, x_train, y_train):
    return model.fit(x_train, y_train)

def all_models(x_train, x_test, y_train, y_test):
    objects = [LogisticRegression(fit_intercept=False, C=1e12), Pipeline([('ss', StandardScaler()), ('knn', KNeighborsClassifier())]), 
               GaussianNB(), DecisionTreeClassifier(), RandomForestClassifier(), XGBClassifier(use_label_encoder=False), 
               Pipeline([('ss', StandardScaler()), ('svm', SVC())])]
    models = []
    precision = []
    recall = []
    accuracy = []
    f1 = []
    index = ['Logistic Regression', 'K-Nearest Neighbors', 'Naive Bayes', 'Decision Tree', 
             'Random Forest', 'XGBoost', 'Support Vector Machine']

    for i, o in enumerate(objects):
        print(f'{index[i]} Results:')
        models.append(make_model(o, x_train, y_train))
        metrics(models[-1], x_test, y_test)
        
    for i in models:
        prediction = i.predict(x_test)
        precision.append(precision_score(y_test, prediction, average='micro'))
        recall.append(recall_score(y_test, prediction, average='micro'))
        accuracy.append(accuracy_score(y_test, prediction))
        f1.append(f1_score(y_test, prediction, average='micro'))
    df = pd.DataFrame(np.array([precision, recall, accuracy, f1]).T, 
                      index = index, columns = ['Precision Score', 'Recall Score', 'Accuracy Score', 'F1 Score']).style.format('{:.2%}')
    display(df)
    print(f'The model with the highest precision score is {df.data.idxmax()[0]}.')
    print(f'The model with the highest recall score is {df.data.idxmax()[1]}.')
    print(f'The model with the highest accuracy score is {df.data.idxmax()[2]}.')
    print(f'The model with the highest F1 score is {df.data.idxmax()[3]}.')

    return models