import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score



def cleanData(training, testing):
    training['Work Experience in Current Job [years]'] = training['Work Experience in Current Job [years]'].replace({'#NUM!': training['Work Experience in Current Job [years]'].mode()})
    testing['Work Experience in Current Job [years]'] = testing['Work Experience in Current Job [years]'].replace({'#NUM!':training['Work Experience in Current Job [years]'].mode()})

    training['Gender'] = training['Gender'].replace({'f':'female'})
    #training['Gender'] = training['Gender'].replace({'0':'other'})
    #training['Gender'] = training['Gender'].replace({'unknow':'other'})
    testing['Gender'] = testing['Gender'].replace({'f':'female'})
    #testing['Gender'] = testing['Gender'].replace({'0':'other'})
    #testing['Gender'] = testing['Gender'].replace({'unknow':'other'})
    #training['University Degree'] = training['University Degree'].replace({'0':'No'})
    #testing['University Degree'] = testing['University Degree'].replace({'0':'No'})
    #training['Hair Color'] = training['Hair Color'].replace({'0':'Other'})
    #training['Hair Color'] = training['Hair Color'].replace({'Unknown':'Other'})
    #testing['Hair Color'] = testing['Hair Color'].replace({'0':'Other'})
    #testing['Hair Color'] = testing['Hair Color'].replace({'Unknown':'Other'})
    #training['Housing Situation'] = training['Housing Situation'].replace({'nA':training['Satisfaction with employer'].mode()})
    #testing['Housing Situation'] = testing['Housing Situation'].replace({'nA': training['Satisfaction with employer'].mode()})
    training['Yearly Income in addition to Salary (e.g. Rental Income)'] = training['Yearly Income in addition to Salary (e.g. Rental Income)'].str.strip('EUR')
    testing['Yearly Income in addition to Salary (e.g. Rental Income)'] = testing['Yearly Income in addition to Salary (e.g. Rental Income)'].str.strip('EUR')
    
    naFillers = {
        'Year of Record': training['Year of Record'].median(),
        'Gender': 'female',
        'Housing Situation': training['Housing Situation'].mode(),
        'Satisfaction with employer': training['Satisfaction with employer'].mode(),
        'Profession': 'principal administrative associate',
        'University Degree': 'No',
        'Hair Color': 'Black'}
    data = pd.concat([training, testing], ignore_index=True)
    for i in naFillers.keys():
        data[i] = data[i].fillna(naFillers[i])
    return data

def createCatsNums(data, cats, nums, normalize=True):
    for i,cat in enumerate(cats):
        valueCounts = data[cat].value_counts(dropna=False, normalize=normalize).to_dict()
        nm = cat + '_FE_FULL'
        data[nm] = data[cat].map(valueCounts)
        data[nm] = data[nm].astype('float32')
        for j,num in enumerate(nums):
            newColumn = cat +'_'+ num
            print('timeblock frequency encoding:', newColumn)
            data[newColumn] = data[cat].astype(str)+'_'+data[num].astype(str)
            tempData = data[newColumn]
            encode = tempData.value_counts(normalize=True).to_dict()
            data[newColumn] = data[newColumn].map(encode)
            data[newColumn] = data[newColumn]/data[cat+'_FE_FULL']
    return data

def main():
    training = pd.read_csv('training.csv')
    testing = pd.read_csv('testing.csv')    
    data = cleanData(training, testing) 

    cats = ['Year of Record', 'Gender', 'Country', 'Profession', 'Housing Situation', 'Satisfaction with employer', 'University Degree', 'Wears Glasses', 'Hair Color', 'Age']
    nums = ['Size of City', 'Body Height [cm]', 'Crime Level in the City of Employement', 'Work Experience in Current Job [years]', 'Yearly Income in addition to Salary (e.g. Rental Income)']
    data = createCatsNums(data, cats, nums)

    for i in training.dtypes[training.dtypes == 'object'].index.tolist():
        featureLE = LabelEncoder()
        featureLE.fit(data[i].unique().astype(str))
        data[i] = featureLE.transform(data[i].astype(str))

    delColumn = set(['Total Yearly Income [EUR]','Instance'])
    featuresColumns =  list(set(data) - delColumn)

    XTrain,XTest = data[featuresColumns].iloc[:1048573],data[featuresColumns].iloc[1048574:]
    YTrain = data['Total Yearly Income [EUR]'].iloc[:1048573]
    XTestID = data['Instance'].iloc[1048574:]
    xTrain,xValidation,yTrain,yValidation = train_test_split(XTrain,YTrain,test_size=0.2,random_state=1234)    

    params = {'max_depth': 20, 'learning_rate': 0.001, "boosting": "gbdt", "bagging_seed": 11, "metric": 'mse', "verbosity": -1}

    print('training')
    trainingData = np.array(xTrain)      #label=yTrain
    validationData = np.array(xValidation) #label=yValidation
    #clf = lgb.train(params, trainingData, 200000, valid_sets = [trainingData, validationData], verbose_eval=1000, early_stopping_rounds=500)
    clf = CatBoostRegressor(iterations=500,learning_rate=1,depth=10)
    clf.fit(trainingData, yTrain)

    testingPred = clf.predict(XTest)
    print('done')
    validationPred = clf.predict(xValidation)
    mse = mean_squared_error(yValidation, validationPred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(yValidation, validationPred)
    print(rmse)
    print(mae)
    np.savetxt('results.csv', testingPred)

   
if __name__ == '__main__':
    main()
