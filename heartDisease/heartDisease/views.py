from django.shortcuts import render, redirect
import pandas as pd
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

url = 'https://raw.githubusercontent.com/Mansi-Saini/Project2/main/heart_disease_data.csv'
df_disease = pd.read_csv(url)
# print(df_disease)
df_disease.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                      'ca', 'thal', 'target', 'percentage']
df_disease.dropna(inplace=True)

features = df_disease[
    ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
     'target']].values
df_disease1 = pd.DataFrame(features,
                           columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
                                    'oldpeak', 'slope', 'ca', 'thal', 'target'])
features = df_disease[
    ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
     'percentage']].values
df_disease2 = pd.DataFrame(features,
                           columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
                                    'oldpeak', 'slope', 'ca', 'thal', 'percentage'])

from sklearn.model_selection import train_test_split

X_train1, X_test1, y_train1, y_test1 = train_test_split(df_disease1.drop('target', axis=1), df_disease1['target'],
                                                        test_size=0.25, random_state=6)
from sklearn.model_selection import train_test_split

X_train2, X_test2, y_train2, y_test2 = train_test_split(df_disease2.drop('percentage', axis=1),
                                                        df_disease2['percentage'], test_size=0.25, random_state=2)


def home(request):
    # train
    return render(request, 'home.html')


def result(request):
    regressor = LinearRegression()
    regressor.fit(X_train2, y_train2)
    model = LogisticRegression()
    model.fit(X_train1, y_train1)
    # age, sex, cp, trestbps, chol,
    # fbs, restecg, thalach, exang, oldpeak, slope, ca and thal
    a = float(request.GET.get('age', '0'))
    gm = request.GET.get('male', "off")
    gf = request.GET.get('female', "off")
    if gm:
        g = 0.0
    elif gf:
        g = 1.0
    cp = float(request.GET.get('cp', '0'))
    t = float(request.GET.get('trestbps', '0'))
    c = float(request.GET.get('chol', '0'))
    fbs = float(request.GET.get('fbs'))
    re = float(request.GET.get('restecg'))
    th = float(request.GET.get('thalach'))
    ex = float(request.GET.get('exang', '0'))
    old = float(request.GET.get('oldpeak', '0'))
    s = float(request.GET.get('slope', '0'))
    ca = float(request.GET.get('ca', '0'))
    thal = float(request.GET.get('thal', '0'))

    features = [a, g, cp, t, c, fbs, re, th, ex, old, s, ca, thal]
    df = pd.DataFrame([features],
                            columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',])
    target = model.predict(df)
    per = regressor.predict(df)
    if target :
        res = "You have heart disease"
        res2 = ''
    else:
        res = "You don't have heart disease"
        res2 = f", Chances of you having heart disease is {round(per[0], 2)}"
    result2 = {'percent': res2, 'tar': res}
    return render(request, 'home.html', {"result2": result2})