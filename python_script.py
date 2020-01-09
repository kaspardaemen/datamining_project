import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import missingno as msno
from scipy.stats import binom_test
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import validation_curve

np.random.seed(100) 

# =============================================================================
# FUNCTIONS
# =============================================================================

def neirest_neighbors(X_test,y_test):
    nn_model.fit(X_train, y_train)
    return  nn_model.predict(X_test)[0]

def random_forest(X_test,y_test):
    rf_model.fit(X_train, y_train)
    return  rf_model.predict(X_test)[0]

def decision_tree(X_test,y_test):
    dt_model.fit(X_train, y_train)
    return  dt_model.predict(X_test)[0]


def accuracy_per_class(conf_matrix):
    result = {}
    #i = true label, j = predicted label
    for i in range(0,len(conf_matrix)): 
        print(i)
        right = []
        total = []
        for j in range(0,len(conf_matrix[i])):
            total.append(conf_matrix[i,j])
            if (i == j):
                right.append(conf_matrix[i,j])
        result[i] = {}  
        result[i]['name'] = classes[i]
        result[i]['right'] = sum(right)
        result[i]['total'] = sum(total)
        result[i]['accuracy'] = sum(right)/sum(total)
    return result


def evaluate_predictions(results):
    for key, value in results.items():
        results[key]['accuracy'] = accuracy_score(y, results[key]['labels'])
        results[key]['conf_matrix'] = confusion_matrix(y, results[key]['labels'])
        results[key]['accuracy_per_class'] = accuracy_per_class(results[key]['conf_matrix']) 


def sign_test(class1, class2, y, one_sided):
    class_1_pred = results[class1]['labels']
    class_2_pred = results[class2]['labels']

    
    c_c = len([1 for x in zip(class_1_pred, class_2_pred, y) if x[0] == x[2] and x[1] == x[2]])
    c_w = len([1 for x in zip(class_1_pred, class_2_pred, y) if x[0] == x[2] and x[1] != x[2]])
    w_c = len([1 for x in zip(class_1_pred, class_2_pred, y) if x[0] != x[2] and x[1] == x[2]])
    w_w = len([1 for x in zip(class_1_pred, class_2_pred, y) if x[0] != x[2] and x[1] != x[2]])
    
    table = np.array([[c_c,c_w],[w_c,w_w]])
    
    print("The wrong/correct 2x2 table: \n",table)
    print('\nThe first classifier won {} times, while the second classifier won {} times. '.format(c_w,w_c))
    
    N = w_c + c_w
    b = binom(N,0.5)

    p_value = (1-b.cdf(max(w_c,c_w)-1)) if one_sided else b.cdf(min(w_c,c_w)) + (1-b.cdf(max(w_c,c_w)-1))
   
    
    print('Null hypothesis: It is equally likely that one classifier is performing better than the other')
    if (one_sided):
        if (c_w == w_c):
            print("both classifiers have performed equally")
            return
        best_class = class1 if c_w > w_c else class2
        worst_class = class1 if c_w < w_c else class2
        print('We are testing if {} performed significantly better than {}'.format(best_class,worst_class))
    
    print('The p-value is {:.3f}'.format(p_value))
    return table
    
def plot_validation_curve(model,X,y,param_name,cv,model_name, start, stop, step):
    param_range = np.arange(start,stop,step)
    train_scores, test_scores = validation_curve(
                                model,
                                X = X, y = y,
                                param_name = param_name,
                                param_range = param_range,
                                cv=cv, #number of folds of the (stratified) cross validation
                                )
    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    plt.plot(param_range, train_mean, "-", label="Training score" )
    plt.plot(param_range, test_mean, "-", label="Test score" )
    
    # Create plot
    plt.title('Validation Curve With {}'.format(model_name))
    plt.xlabel(param_name)
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show()
    
    #pritn the max test score
    first_highest = np.where(test_mean == np.amax(test_mean))
    print("The max accuracy is: " , np.amax(test_mean) , "for a value of", (first_highest[0][0] * step) + 1 )

# =============================================================================
# IMPORTING THE DATA
# =============================================================================

class_df = pd.read_csv('class.csv')
classes = class_df['Class_Type'].values.tolist()
zoo_df = pd.read_csv('zoo.csv')


# =============================================================================
# PREPROCESSING
# =============================================================================
data = zoo_df.drop(['animal_name'], axis=1)

agg_data = data.groupby('class_type').mean().reset_index()

for i in range(1, len(classes)+1):
    agg_data['class_type'].replace(i,classes[i-1], inplace=True)

#msno.matrix(data)
#sns.boxplot(data['legs'])
#sns.distplot(data['legs'])
 
X = data.drop('class_type', 1).values
y = data['class_type'].values

results = {}

# =============================================================================
# Tuning the hyperparameters
# =============================================================================

##RF
#plot_validation_curve(RandomForestClassifier(max_depth=18),
#                      X=X, y=y, 
#                      param_name = 'n_estimators',
#                      start=1,stop=100,step = 5,
#                      cv=4,
#                      model_name='Random Forest')

##KNN
#plot_validation_curve(KNeighborsClassifier(),
#                        X = X, y = y,
#                        param_name = 'n_neighbors',
#                        start= 1,stop= 75, step= 1,
#                        cv=4,
#                        model_name = 'K-neirest neighbors'
#                        )
#
#plot_validation_curve(KNeighborsClassifier(),
#                        X = X, y = y,
#                        param_name = 'p',
#                        start= 1,stop= 100, step= 1,
#                        cv=4,
#                        model_name = 'K-neirest neighbors'
#                        )

#DT
plot_validation_curve(DecisionTreeClassifier(),
                      X=X, y=y, 
                      param_name = 'max_depth',
                      start=1,stop=100,step = 1,
                      cv=4,
                      model_name='Decision Tree')

# =============================================================================
# RANDOM FOREST
# =============================================================================

rf_model = RandomForestClassifier(n_estimators = 16, max_depth=18)

# =============================================================================
# NEIREST NEIGHBORS
# =============================================================================

nn_model = KNeighborsClassifier(n_neighbors =1, p=2)

# =============================================================================
# DECISION TREE    
# =============================================================================

dt_model = DecisionTreeClassifier(max_depth = 15, criterion="gini")
    
# =============================================================================
# TRAINING THE MODELS
# =============================================================================

loo = LeaveOneOut()

rf_labels = []
nn_labels = []
dt_labels = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    rf_label = random_forest(X_test, y_test)
    rf_labels.append(rf_label)
    
    nn_label = neirest_neighbors(X_test, y_test)
    nn_labels.append(nn_label)
    
    dt_label = decision_tree(X_test, y_test)
    dt_labels.append(dt_label) 
    
    
results['rf'] = {}
results['rf']['labels'] = np.array(rf_labels)
results['nn'] = {}
results['nn']['labels'] = np.array(nn_labels)
results['dt'] = {}
results['dt']['labels'] = np.array(dt_labels)

# =============================================================================
# Evaluation
# =============================================================================

evaluate_predictions(results)

# =============================================================================
# SIGN TEST
# =============================================================================
from scipy.stats import binom

wrong_corret_table = sign_test('nn', 'dt', y, False)






