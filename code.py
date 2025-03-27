### Kaggle - Binary Classification of Insurance Cross Selling
### https://www.kaggle.com/competitions/playground-series-s4e7/overview

path = 'input your path here'
path_data = path + '/Data/'
path_results = path + '/Results/'

# Libraries
if True:
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_curve, accuracy_score, precision_score, f1_score, recall_score, classification_report, roc_curve, auc, roc_auc_score
    import matplotlib.pyplot as plt
    from xgboost import cv
    import xgboost as xgb
    import time
    import seaborn as sns
    
def roc_auc_curve_sim(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba) 
    roc_auc = auc(fpr, tpr)
    # Plot the ROC curve
    plt.clf()
    plt.figure()  
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

# Import data - rename the target column as 'Target' - encode the target if needed (classification)
if True:
    print('Importing data')
    tab = pd.read_csv(path_data + 'train.csv')
    test = pd.read_csv(path_data + 'test.csv')
    target = 'Response'
    tab.rename(columns = {target: 'Target'}, inplace = True)
    if True:
        tab = tab.sample(frac = 1) # 0.02
        test = test.head(1000)
    
# Explore target mean by categorical values + their distribution
if False:
    for col in ['Gender', 'Vehicle_Age', 'Vehicle_Damage',
                'Driving_License', 'Previously_Insured',]:
        res = tab.groupby(col).agg({'id': lambda x : x.size,
                                    'Target': np.mean}).reset_index()
        print(res)
        
    for col in ['Region_Code', 'Policy_Sales_Channel', 'Vintage', 'Age']:
        res = tab.groupby(col).agg({'id': lambda x : x.size,
                                    'Target': np.mean}).reset_index() 
        res['id prop'] = res['id']/res['id'].sum()
        plt.clf()
        plt.figure()
        plt.plot(res[col], res['Target'], label='Target Mean', color = 'red')
        plt.plot(res[col], res['id prop'], label='Proportion', color = 'blue')
        plt.legend()
    
if False:
    # explore the impact of Vintage
    def func(x):
        return int(x/20)
    tab['Vintage func'] = tab['Vintage'].apply(lambda x : func(x))
    res = tab.groupby('Vintage func').agg({'id': lambda x : x.size,
                                           'Target': np.mean}).reset_index()
    print(res)
    plt.clf()
    plt.figure()
    plt.plot(res['Vintage func'], res['Target'], label='Target Mean', color = 'red')
    plt.legend()
    tab.drop(columns = ['Vintage func'], inplace = True)

# Tratar categorical variables
if True:
    # Gender 01, Vehicle_Age 012, Vehicle_Damage 01
    encoder = {'Male': 0, 'Female': 1}
    tab['Gender'] = tab['Gender'].apply(lambda x: encoder[x])
    test['Gender'] = test['Gender'].apply(lambda x: encoder[x])
    encoder = {'No': 0, 'Yes': 1}
    tab['Vehicle_Damage'] = tab['Vehicle_Damage'].apply(lambda x: encoder[x])
    test['Vehicle_Damage'] = test['Vehicle_Damage'].apply(lambda x: encoder[x])
    encoder = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
    tab['Vehicle_Age'] = tab['Vehicle_Age'].apply(lambda x: encoder[x])
    test['Vehicle_Age'] = test['Vehicle_Age'].apply(lambda x: encoder[x])

    tab['Region_Code'] = tab['Region_Code'].apply(lambda x : int(x))
    test['Region_Code'] = test['Region_Code'].apply(lambda x : int(x))

    # Age
    def function_age(x):
        if x <= 26:
            return 80
        if x <= 32:
            return 50 + 5* (32 - x)
        if x >= 80:
            return 80
        return x
    tab['Age'] = tab['Age'].apply(lambda x : function_age(x))
    test['Age'] = test['Age'].apply(lambda x : function_age(x))

    # Region_Code --> reorder per target mean
    # Policy_Sales_Channel --> group the small ones in to other, and reorder per target mean
    
    for col in [
            #'Policy_Sales_Channel'
            ]:
        # Group all ... with < 1000 people as 'Others'
        res = tab.groupby([col]).size().reset_index()
        res['New ' + col] = res[col].apply(lambda x : str(x))
        res.loc[res[0] <= 1000, 'New ' + col] = -10
        tab = tab.merge(res[[col, 'New ' + col]], on = col, how = 'left')
        tab.drop(columns = [col], inplace = True)
        tab.rename(columns = {'New ' + col: col}, inplace = True)
        # Apply to test as well
        test = test.merge(res[[col, 'New ' + col]], on = col, how = 'left')
        test.drop(columns = [col], inplace = True)
        test.rename(columns = {'New ' + col: col}, inplace = True)
    
    for col in [
            #'Policy_Sales_Channel'
            ]:
        res = tab.groupby(col)['Target'].mean().reset_index().sort_values(['Target'], ascending = False).reset_index()
        res['New ' + col] = res.index
        tab = tab.merge(res[[col, 'New ' + col]], on = col, how = 'left')
        tab.drop(columns = [col], inplace = True)
        tab.rename(columns = {'New ' + col: col}, inplace = True)
        # Apply to test as well
        test = test.merge(res[[col, 'New ' + col]], on = col, how = 'left')
        test.drop(columns = [col], inplace = True)
        test.rename(columns = {'New ' + col: col}, inplace = True)
    
    # Region_Code : create categories depending on the target proportion
    num = {'Region_Code': 20, 'Policy_Sales_Channel': 20, 'Vintage': 20}
    for col in ['Region_Code', 'Policy_Sales_Channel', 'Vintage']:
        print(col)
        print('Qtd of old columns:', tab[col].nunique())
        res = tab.groupby(col).agg({'Target' : np.mean,
                                    'id': lambda x : x.nunique()}).reset_index().sort_values(['Target'], ascending = False).reset_index()
        standard_dev = np.std(res['Target'])/num[col] # defines the quantity of categories
        maxi = res['Target'].max()
        def determine_cat(x):
            i = 0
            while x <= maxi - i * standard_dev:
                i += 1
            return i
        res[col + ' cat'] = res['Target'].apply(lambda x : determine_cat(x))
        tab = tab.merge(res[[col, col + ' cat']], on = col, how = 'left')
        tab.drop(columns = [col], inplace = True)
        tab.rename(columns = {col + ' cat' : col}, inplace = True)
        print('Qtd of new columns:', tab[col].nunique())
        print()
        
        

    # Anual Premium
    if False:
        tab['Annual_Premium_Cat'] = tab['Annual_Premium'].apply(lambda x : round((x-2.630000e+03)/(5.401650e+05 -  2.630000e+03), 1))
        col = 'Annual_Premium_Cat'
        res = tab.groupby(col).agg({'id': lambda x : x.size,
                                        'Target': np.mean}).reset_index()
        res['id prop'] = res['id']/res['id'].sum()
        plt.clf()
        plt.figure()
        plt.plot(res[col], res['Target'], label='Target Mean', color = 'red')
        plt.plot(res[col], res['id prop'], label='Proportion', color = 'blue')
        plt.legend() 


if True:
    # Turn multi-class categorical features into dummy variables
    col_add = []
    if True:
        col_add = [
            'Policy_Sales_Channel', 
            'Region_Code', 
            'Vintage'
            ]
    for col in [
            'Vehicle_Age', 
            ] + col_add:
        for value in tab[col].unique().tolist()[:-1]:
            tab[col + ' - ' + str(value)] = 0
            tab.loc[tab[col] == value, col + ' - ' + str(value)] = 1
            test[col + ' - ' + str(value)] = 0
            test.loc[test[col] == value, col + ' - ' + str(value)] = 1
        tab.drop(columns = [col], inplace = True)   
        test.drop(columns = [col], inplace = True)

# Correlation matrix
if False:
    print('Correlation matrix.')
    matrix = tab.corr()
    sns.heatmap(matrix)

print('Feature selection, and data preparation.')
if True:
    features = [i for i in tab.columns if i not in ['Target', 'id']]
    x, y = tab[features], tab['Target']
    del tab
    
    # standard scaler
    if False:
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)
        submissions = scaler.transform(test[features])
    
    if True:
        print('Scaling à la main')
        res = pd.DataFrame(columns  = ['Mean', 'Std'])
        for ccc in x.columns:
            res.loc[ccc, 'Mean'] = x[ccc].mean()
            res.loc[ccc, 'Std'] = np.std(x[ccc])
        for ccc in x.columns:
            print(ccc)
            m, std = res.loc[ccc, 'Mean'], res.loc[ccc, 'Std']
            x[ccc] = x[ccc].apply(lambda x : (x - m) / std)

    del test
    
    # train test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 0)
    del x, y

    print('Proportions:', y_train.value_counts(1))
    
   





''' ---- Models ---- '''
print('--- Models ---')

# Gradient Boosting
if False:
    print('First xgb')
    t1 = time.time()
    xgb_classifier = XGBClassifier(n_estimators = 300, max_depth = 8, learning_rate = 0.03, subsample = 0.7)
    xgb_classifier.fit(x_train, y_train)
    print('Train ROC AUC:', roc_auc_score(y_train, xgb_classifier.predict_proba(x_train)[:, 1]))
    print('Test ROC AUC:', roc_auc_score(y_test, xgb_classifier.predict_proba(x_test)[:, 1]))
    roc_auc_curve_sim(y_test, xgb_classifier.predict_proba(x_test)[:, 1])
    t2 = time.time()
    print('Model time:', round(t2 - t1), 'seconds')
    
    t1 = time.time()
    param_grid = {
        'n_estimators': [500, 550, 600, 650], #[600, 650, 700],
        'max_depth': [5, 6, 7, 8, 9, 10], #[6, 8, 10],
        'learning_rate': [0.01, 0.015, 0.02], #[0.03, 0.04, 0.05, 0.06],
        'subsample': [0.75], #[0.7, 0.75],
        'colsample_bytree': [0.5],  #[0.4, 0.5, 0.6],
        'min_child_weight': [14, 16, 18, 20], #[1, 3, 5, 7, 9, 10, 12],
        }
    
    # Grid Search
    if False:
        xgb_classifier = XGBClassifier()
        print('Grid Search com criteria accuracy')
        grid_search = GridSearchCV(xgb_classifier, param_grid, cv=5, scoring='accuracy', n_jobs = 5)
        grid_search.fit(x_train, y_train)
        res = pd.DataFrame(grid_search.cv_results_)
        t2 = time.time()
        print('Time to the gridsearch:', round((t2 - t1)/60, 1), 'min')
        print('Best hyperparameters: ', grid_search.best_params_)
        print('Best accuracy score:', round(grid_search.best_score_, 4))
        xgb_classifier = grid_search.best_estimator_

        bestparams = grid_search.best_params_
        max_depth = bestparams['max_depth']
        learning_rate = bestparams['learning_rate']
        n_estimators = bestparams['n_estimators']
        subsample = bestparams['subsample']   
        
    # parameters after gridsearch
    print('xgb with tuning')
    n_estimators = 1000 # 1000
    colsample_bytree = 0.5 # 0.5
    learning_rate = 0.018 # 0.018
    max_depth = 15 # 15
    min_child_weight = 10 # 10
    subsample = 0.5 # 0.75

    xgb_classifier = XGBClassifier(n_estimators = n_estimators, max_depth = max_depth, learning_rate = learning_rate, subsample = subsample,
                                   min_child_weight = min_child_weight, colsample_bytree = colsample_bytree)
    xgb_classifier.fit(x_train, y_train)
    print('Train ROC AUC:', roc_auc_score(y_train, xgb_classifier.predict_proba(x_train)[:, 1]))
    print('Test ROC AUC:', roc_auc_score(y_test, xgb_classifier.predict_proba(x_test)[:, 1]))
    feat = xgb_classifier.feature_importances_



    # Submission
    if False:
        xgb_classifier = XGBClassifier(n_estimators = n_estimators, max_depth = max_depth, learning_rate = learning_rate, subsample = subsample,
                                       min_child_weight = min_child_weight, colsample_bytree = colsample_bytree)
        new_x = np.concatenate((x_train, x_test), axis = 0)
        new_y = np.concatenate((y_train, y_test), axis = 0)
        xgb_classifier.fit(new_x, new_y)
        sub_preds = xgb_classifier.predict_proba(submissions)[:, 1]
        res = pd.DataFrame()
        res['id'] = test['id'].tolist()
        res['Target'] = sub_preds.tolist()
        res.to_csv(path_results + 'Model - xgb with tuning.csv', index = False)
        




'''
Tests on 0.005 of the dataset (0.5%) with 0.25 of test_split

Best hyperparameters:  {'colsample_bytree': 0.5, 'learning_rate': 0.015, 'max_depth': 7, 'min_child_weight': 14, 'n_estimators': 600, 'subsample': 0.75}
Best accuracy score: 0.878

Best hyperparameters:  {'colsample_bytree': 0.5, 'learning_rate': 0.015, 'max_depth': 9, 'min_child_weight': 14, 'n_estimators': 500, 'subsample': 0.75}
Best accuracy score: 0.8784

Tests on 0.02 of the dataset (2%) with 0.40 of test_split:

Best hyperparameters:  {'colsample_bytree': 0.5, 'learning_rate': 0.02, 'max_depth': 7, 'min_child_weight': 18, 'n_estimators': 600, 'subsample': 0.75}
Best accuracy score: 0.879


'''




# Neural Network
if True:
    import keras
    import keras_tuner
    from keras.models import Sequential
    from keras.layers import Dense, Input
    from keras.utils import to_categorical

    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim = x_train.shape[1]))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer = opt, 
                  loss = 'binary_crossentropy', 
                  metrics = [keras.metrics.AUC()])
    model.fit(x_train, y_train, 
              epochs = 20,
              batch_size = 1000,
              verbose = 0
              )
    print('Train ROC AUC:', roc_auc_score(y_train, model.predict(x_train, verbose = 0)))
    print('Test ROC AUC:', roc_auc_score(y_test, model.predict(x_test, verbose = 0)))


    # Grid Search
    if True:
        print('Grid Search...')
        size_dense_1 = [128]
        size_dense_2 = [10, 20]
        learning_rate = [0.011, 0.01]
        batch = [1000, 2000]
        
        res = pd.DataFrame(columns = ['Size Dense 1', 'Size Dense 2', 'Learning Rate', 'Batch Size'])
        i, taille_total = 0, len(size_dense_1) * len(size_dense_2) * len(learning_rate) * len(batch)
        for s1 in size_dense_1:
            for s2 in size_dense_2:
                for lr in learning_rate:
                    for b in batch:
                        i+=1
                        t0 = time.time()
                        model = Sequential()
                        model.add(Dense(s1, activation='relu', input_dim = x_train.shape[1]))
                        model.add(Dense(s2, activation='relu'))
                        model.add(Dense(1, activation='sigmoid'))
                        opt = keras.optimizers.Adam(learning_rate = lr)
                        model.compile(optimizer = opt, 
                                      loss = 'binary_crossentropy', 
                                      metrics = [keras.metrics.AUC()])
                        model.fit(x_train, y_train, 
                                  epochs = 10,
                                  batch_size = b,
                                  verbose = 0
                                  )
                        res.loc[i, 'Size Dense 1'], res.loc[i, 'Size Dense 2'], res.loc[i, 'Learning Rate'], res.loc[i, 'Batch Size'] = s1, s2, lr, b
                        res.loc[i, 'Train AUC'] =  roc_auc_score(y_train, model.predict(x_train, verbose = 0))
                        res.loc[i, 'Test AUC'] =  roc_auc_score(y_test, model.predict(x_test, verbose = 0))
                        t1 = time.time()
                        print(s1, s2, lr, b, '% of total done:', round(100 * i/taille_total, 1), '% - estimated remaining time:', round((t1 - t0) * (taille_total - i) / (60), 1), 'minutes')














