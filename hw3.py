from argparse import ArgumentParser
from hashlib import new
from inspect import trace
from lib2to3.pytree import Node
from msilib.schema import IniLocator
from typing import Tuple, Union, List, Any
import numpy as np
import pandas as pd
from pkg_resources import resource_listdir

from sklearn.linear_model import LogisticRegression
def data_preprocessing(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return the preprocessed data (X_train, X_test, y_train, y_test). 
    You will need to remove the "Email No." column since it is not a valid feature.
    """
    train_goal = int(data.shape[0]*0.8)
    y_train_temp = data[:train_goal]
    y_test_temp = data[train_goal:]
    y_train = y_train_temp.loc[:train_goal, ["Prediction"]]
    y_test = y_test_temp.loc[train_goal:, ["Prediction"]]
    data = data.drop(["Prediction", "Email No."], axis=1)
    x_train = data.iloc[:train_goal, :]
    x_test = data.iloc[train_goal:, :]
    return (x_train, x_test, y_train, y_test)
    #raise NotImplementedError

class Node:
    def __init__(self) -> None:
        self.left = None       #left
        self.right = None       #right
        self.threshold = None
        self.result = None

class DecisionTree:
    "Add more of your code here if you want to"
    def __init__(self) -> None:
        self.tree = None
    def confusion(self, a: int, b: int) -> int:
        if (a+b == 0):  return 0
        else:
            return (1- ((a/(a+b))**2) - ((b/(a+b))**2))
    def total(self, c: int , d: int, e: int, f:int ) -> int:
        return ((c+d)/(c+d+e+f)*self.confusion(c,d)+(e+f)/(c+d+e+f)*self.confusion(e,f))
    def choose_threshold(self, X: pd.DataFrame, y: pd.DataFrame) -> tuple:
        ''' 
            return -1 when all of y are -1
            return -2 when all of y are 1
            otherwise return the best threshold
        '''
        final_threshold = float("inf") ; final_comparenum = float("inf") ; final_id = 0
        for i in range(X.shape[1]):
            l = []
            for j in range(X.shape[0]):
                l.append((X.iloc[j, i], j))
            l.sort()
            rt = 0 ; rf = 0 ; lt = 0 ; lf = 0
            for j in range(len(l)):
                if (y.iloc[l[j][1],0] == 1):      rt += 1
                elif (y.iloc[l[j][1],0] == -1):   rf += 1
            if (rt == 0):
                return (-1, -1)
            elif (rf == 0):
                return (-2, -2)
            for j in range(len(l)-1):
                if (y.iloc[l[j][1],0] == 1):
                    rt -= 1
                    lt += 1
                else:
                    rf -= 1
                    lf += 1
                if (l[j][0] != l[j+1][0]):
                    temp = self.total(lt, lf, rt, rf)
                    if (temp < final_comparenum):
                        final_comparenum = temp
                        final_threshold = (l[j][0]+l[j+1][0])/2
                        final_id = i
        #print((final_threshold, final_id))
        return (final_threshold, final_id)

    def set_tree(self, X: pd.DataFrame, y: pd.DataFrame) -> Node:
        #   stop condition
        new_node = Node()
        thres = self.choose_threshold(X, y)
        if (thres == (-1, -1)):
            new_node.result = -1
            return new_node
        elif (thres == (-2, -2)):
            new_node.result = 1
            return new_node
        #   recursion starts from here
        l_list = [] ; r_list = []
        for i in range(X.shape[0]):
            if (X.iloc[i, thres[1]] < thres[0]):
                l_list.append(i)
            else:
                r_list.append(i)
        l_x = X.iloc[l_list, :]
        l_y = y.iloc[l_list, :]
        r_x = X.iloc[r_list, :]
        r_y = y.iloc[r_list, :]
        new_node.threshold = thres
        new_node.left = self.set_tree(l_x, l_y)
        new_node.right = self.set_tree(r_x, r_y)
        return new_node

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        "Fit the model with training data"
        self.tree = self.set_tree(X, y)
        return 
        #raise NotImplementedError

    def search(self, x: pd.DataFrame, i: int, node: Node) -> int:
        if (node.result is not None):
            return node.result
        if (x.iloc[i, node.threshold[1]] < node.threshold[0]):
            return self.search(x, i ,node.left)
        else:
            return self.search(x, i ,node.right)

    def predict(self, X: pd.DataFrame) -> Any:
        "Make predictions for the testing data"
        y = []
        for i in range(X.shape[0]):
            y.append(self.search(X, i, self.tree))
        return y
        #raise NotImplementedError

class RandomForest:
    "Add more of your code here if you want to"
    def __init__(self, seed: int = 42, num_trees: int = 5):
        self.num_trees = num_trees
        self.forest = None
        np.random.seed(seed)

    def bagging(self, X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        "DO NOT modify this function. This function is deliberately given to make your result reproducible."
        index = np.random.randint(0, X.shape[0], int(X.shape[0] / 2))
        return X.iloc[index, :], y.iloc[index]

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        roots = []
        for i in range(self.num_trees):
            new_data = self.bagging(X, y)
            new_root = DecisionTree()
            new_root.fit(new_data[0], new_data[1])
            roots.append(new_root)
        self.forest = roots
        return 
        #raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> Any:
        y = []
        votes = np.zeros(X.shape[0])
        for j in range(len(self.forest)):
            result = np.array(self.forest[j].predict(X))
            votes = votes + result
        for i in range(len(votes)):
            if (votes[i] < 0):
                y.append(-1)
            else:
                y.append(1)        
        return y
        #raise NotImplementedError

def accuracy_score(y_pred: Any, y_label: Any) -> float:
    """
    y_pred: (1d array-like) your prediction
    y_label: (1d array-like) the groundtruth label
    Return the accuracy score
    """
    TP = 0 ; FP = 0 ; TN = 0 ; FN = 0
    for i in range(len(y_pred)):
        if (y_pred[i] == 1 and y_label.iloc[i, 0] == 1):
            TP += 1
        elif (y_pred[i] == 1 and y_label.iloc[i, 0] == -1):
            FP += 1
        elif (y_pred[i] == -1 and y_label.iloc[i, 0] == -1):
            TN += 1
        elif (y_pred[i] == -1 and y_label.iloc[i, 0] == 1):
            FN += 1
    return (TP+TN)/(TP+TN+FP+FN)

def f1_score(y_pred: Any, y_label: Any) -> float:
    """
    y_pred: (1d array-like) your prediction
    y_label: (1d array-like) the groundtruth label
    Return the F1 score
    """
    TP = 0 ; FP = 0 ; TN = 0 ; FN = 0
    for i in range(len(y_pred)):
        if (y_pred[i] == 1 and y_label.iloc[i, 0] == 1):
            TP += 1
        elif (y_pred[i] == 1 and y_label.iloc[i, 0] == -1):
            FP += 1
        elif (y_pred[i] == -1 and y_label.iloc[i, 0] == -1):
            TN += 1
        elif (y_pred[i] == -1 and y_label.iloc[i, 0] == 1):
            FN += 1
    RECALL = TP/(TP+FN)
    Precision = TP/(TP+FP)
    return 2*(Precision*RECALL/(Precision+RECALL))

def cross_validation(model: Union[LogisticRegression, DecisionTree, RandomForest], X: pd.DataFrame, y: pd.DataFrame, folds: int = 5) -> Tuple[float, float]:
    """
    Test the generalizability of the model with 5-fold cross validation
    Return the mean accuracy and F1 score
    """
    gap = int(X.shape[0]/folds)
    acry = [] ; f1s = []
    for i in range(folds):
        X_test = X.iloc[gap*i:gap*(i+1), :]
        y_test = y.iloc[gap*i:gap*(i+1), :]
        x_train1 = X.iloc[:gap*i, :] ; x_train2 = X.iloc[gap*(i+1):, :]
        x_train = pd.concat([x_train1, x_train2])
        y_train1 = y.iloc[:gap*i, :] ; y_train2 = y.iloc[gap*(i+1):, :]
        y_train = pd.concat([y_train1, y_train2])
        model.fit(x_train, y_train)
        acr_temp = accuracy_score(model.predict(X_test), y_test)
        f1_temp = f1_score(model.predict(X_test), y_test) 
        acry.append(acr_temp)
        f1s.append(f1_temp)
    return (sum(acry)/len(acry), sum(f1s)/len(f1s))
    #raise NotImplementedError

def tune_random_forest(choices: List[int], X: pd.DataFrame, y: pd.DataFrame) -> int:
    """
    choices: List of candidates for the number of decision trees in the random forest
    Return the best choice
    """
    best_trees = 0 ; best_f1 = 0
    for j in choices:
        new_forest = RandomForest(num_trees=j)
        scores = cross_validation(new_forest, X, y, 5)
        if (scores[1] > best_f1):
            best_f1 = scores[1]
            best_trees = j
    return best_trees

    #raise NotImplementedError

def main(args):
    """
    This function is provided as a head start
    TA will use his own main function at test time.
    """
    data = pd.read_csv(args.data_path)
    print(data.head())
    print(data['Prediction'].value_counts())
    X_train, X_test, y_train, y_test = data_preprocessing(data)
    
    logistic_regression = LogisticRegression(solver='liblinear', max_iter=500)
    decision_tree = DecisionTree()
    random_forest = RandomForest()
    models = [logistic_regression, decision_tree, random_forest]

    best_f1, best_model = -1, None
    for model in models:
        accuracy, f1 = cross_validation(model, X_train, y_train, 5)
        print(accuracy, f1)
        if f1 > best_f1:
            best_f1, best_model = f1, model
    
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print(accuracy_score(y_pred, y_test), f1_score(y_pred, y_test))
    print(tune_random_forest([5, 11, 17], X_train, y_train))

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./emails.csv')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(parse_arguments())