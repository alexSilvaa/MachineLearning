import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from collections import Counter

from sklearn import svm
from sklearn import tree
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, BayesianRidge, SGDRegressor, LassoLars
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

sns.set(style='white', context='notebook', palette='deep')


def columnDrop(dataset, category):
    dataset = dataset.drop(columns = category)
    return dataset

def oneHotEncode(dataset, category):
    category_dataset = pd.get_dummies(dataset[category], prefix=category)
    dataset = pd.concat([dataset, category_dataset], axis=1)
    dataset = columnDrop(dataset, category)
    return dataset

def normalize(dataset, category):
    dataset[category] = (dataset[category] - dataset[category].min()) / (dataset[category].max() - dataset[category].min())
    return dataset[category]

def z_score(data, category):
    data[category] = (data[category]-data[category].mean())/data[category].std()
    return data[category]

def data_map(data, category, category_, zeros_dataset):
    for row in range(0, len(data)):
        if category_ + str(data.loc[row, category]) in list(zeros_dataset.columns):
            zeros_dataset.loc[row, [category_ + str(data.loc[row, category])]] = 1
    return zeros_dataset

def cleanData(dataset):
    dataset = columnDrop(dataset, 'Instance')
    dataset = columnDrop(dataset, 'Hair Color')
    dataset['Year of Record'] = dataset['Year of Record'].fillna(dataset['Year of Record'].median())
    dataset['Gender'] = dataset['Gender'].fillna(dataset['Gender'].mode())
    dataset['Gender'] = dataset['Gender'].replace({'unknown':dataset['Gender'].mode()})
    dataset['Gender'] = dataset['Gender'].replace({'0':dataset['Gender'].mode()})
    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())
    dataset['Country'] = dataset['Country'].fillna(dataset['Country'].mode())
    dataset['Size of City'] = dataset['Size of City'].fillna(dataset['Size of City'].median())
    dataset['Profession'] = dataset['Profession'].fillna(dataset['Profession'].mode())
    dataset['University Degree'] = dataset['University Degree'].fillna(dataset['University Degree'].mode())
    dataset['University Degree'] = dataset['University Degree'].replace({'0':dataset['University Degree'].mode()})
    dataset = oneHotEncode(dataset, 'Country')
    dataset = oneHotEncode(dataset, 'Gender')
    dataset = oneHotEncode(dataset, 'Profession')
    dataset = oneHotEncode(dataset, 'University Degree')
    dataset['Body Height [cm]'] = z_score(dataset, 'Body Height [cm]')
    dataset['Age'] = z_score(dataset, 'Age')
    dataset['Year of Record'] = z_score(dataset, 'Year of Record')
    dataset['Size of City'] = z_score(dataset, 'Size of City')
    return dataset

def predictionClean(prediction_dataset, zeros_dataset):
    zeros_dataset.columns = [i[0] for i in zeros_dataset.columns]
    prediction_dataset['University Degree'] = prediction_dataset['University Degree'].fillna(prediction_dataset['University Degree'].mode())
    prediction_dataset['University Degree'] = prediction_dataset['University Degree'].replace({'0':prediction_dataset['University Degree'].mode()})
    prediction_dataset['Gender'] = prediction_dataset['Gender'].fillna(prediction_dataset['Gender'].mode())
    prediction_dataset['Gender'] = prediction_dataset['Gender'].replace({'unknown':prediction_dataset['Gender'].mode()})
    prediction_dataset['Gender'] = prediction_dataset['Gender'].replace({'0':prediction_dataset['Gender'].mode()})
    prediction_dataset['Year of Record'] = prediction_dataset['Year of Record'].fillna(prediction_dataset['Year of Record'].median())
    prediction_dataset['Age'] = prediction_dataset['Age'].fillna(prediction_dataset['Age'].median())
    prediction_dataset['Country'] = prediction_dataset['Country'].fillna(prediction_dataset['Country'].mode())
    prediction_dataset['Size of City'] = prediction_dataset['Size of City'].fillna(prediction_dataset['Size of City'].median())
    prediction_dataset['Profession'] = prediction_dataset['Profession'].fillna(prediction_dataset['Profession'].mode())
    zeros_dataset['Body Height [cm]'] = z_score(prediction_dataset, 'Body Height [cm]')
    zeros_dataset['Age'] = z_score(prediction_dataset, 'Age')
    zeros_dataset['Year of Record'] = z_score(prediction_dataset, 'Year of Record')
    zeros_dataset['Size of City'] = z_score(prediction_dataset, 'Size of City')
    zeros_dataset = data_map(prediction_dataset, 'Profession', 'Profession_', zeros_dataset)
    zeros_dataset = data_map(prediction_dataset, 'Country', 'Country_', zeros_dataset)
    zeros_dataset = data_map(prediction_dataset, 'Gender', 'Gender_', zeros_dataset)
    zeros_dataset = data_map(prediction_dataset, 'University Degree', 'University Degree_', zeros_dataset)
    return zeros_dataset

def main():
    dataset = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
    dataset = cleanData(dataset)
    dataset['Income in EUR'] = dataset['Income in EUR'].fillna(dataset['Income in EUR'].median())
    for i in range(0, len(dataset['Income in EUR'])):
        if dataset.loc[i, 'Income in EUR'] < -1000 or dataset.loc[i, 'Income in EUR'] > 3000000:
            dataset.loc[i, 'Income in EUR'] = np.nan
    dataset.dropna(inplace=True)
    
    prediction_dataset = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')

    X = dataset.drop(columns = 'Income in EUR')
    Y = dataset['Income in EUR']

    zeroData = np.zeros((len(prediction_dataset), len(X.columns)))
    zeros_dataset = pd.DataFrame(zeroData, columns=[X.columns])
    prediction_dataset = predictionClean(prediction_dataset, zeros_dataset)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    
    # Regressssion ----------
    regressor = Ridge()
    regressor.fit(X_train, Y_train)
    Y_pred = regressor.predict(X_test)
    Y_pred2 = regressor.predict(prediction_dataset)
    print(np.sqrt(mean_squared_error(Y_test, Y_pred)))
    # --------------------

    
    #np.savetxt('ypred.csv', Y_pred)
    #np.savetxt('ypred2.csv', Y_pred2)
    #print(Y_pred2)
    #np.savetxt('submission data.csv', Y_pred2)
    

   
if __name__ == '__main__':
    main()

