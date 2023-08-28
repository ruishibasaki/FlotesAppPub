import numpy as np
from sklearn.base import BaseEstimator

from sklearn.utils.validation import  check_array, check_is_fitted, check_X_y

'''
Usage:
-----
## Find model with best param 'delta' and 'max_depth' using a GridSearchCV or a StratifiedKFold
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import RegularDecisionTree as rdt
models = [i['class'] for i in instances]
list_nodes = list(G.nodes)
flow_array = [[] for i in range(len(models))]
for i in range(len(models)):
    for n in list_nodes:
        flow_array[i].append(flow[n][i])
nb_sensors = 3
reg_parameters = {'delta': [0.1+0.1*i for i in range(10)], 'max_depth': [4,6,8,10]}
reg_model = rdt.RegularizedDecisionTreeClassifier(max_n_features=nb_sensors)
skf = StratifiedKFold(n_splits=5)
grid_reg = GridSearchCV(reg_model, reg_parameters, cv = skf, n_jobs= 4)
grid_reg.fit(flow_array,models)
print(grid_reg.best_params_)
print(grid_reg.best_score_)
selected_nodes = []
for id_n in list(grid_reg.best_estimator_.selected_features_):
    selected_nodes.append(list_nodes[id_n])
## Feedback: use previously observed nodes
## by keeping the list of INDEXES of the previously selected nodes
hist_observed_nodes = grid_reg.best_estimator_.selected_features_ 
## Then it can be passed as parameter of the fit(.) function
grid_reg.fit(new_flow_array,new_models,hist_observed_nodes)
## WARNING: the order of nodes in flow_array should be the same between each iteration
## of the feedback loop
## Newly selected nodes will not contain elements of hist_observed_node
## You have to keep track of the previously observed nodes indexes.
grid_reg.best_estimator_.selected_features_ 
'''


class Node:
    def __init__(self, depth = 0, counts = None, impurity = None):        
        self.depth = depth        
        self.counts = counts
        self.impurity = impurity
        # links to the left and right child nodes
        self.right = None
        self.left = None        
        # derived from splitting criteria
        self.column = None
        self.threshold = None
        # counts for sample inside the node to belong for each of the given classes
        # depth of the given node
        self.is_terminal = False


    def print(self,tab=''):
        if not self.is_terminal:
            print(f'{tab}X[{self.column}] <= {round(self.threshold,3)} I:{round(self.impurity,4)} {list(self.counts)}')        
            self.left.print(tab+'  ')
            self.right.print(tab+'  ')
        else:
             print(f'{tab}Leaf I:{round(self.impurity,4)} {list(self.counts)}')

class RegularizedDecisionTreeClassifier(BaseEstimator):
    '''
    Usage:
    ------
    import numpy as np
    import RegularDecisionTree as rdt
    nodes_list = list(G.nodes())
    print(len(nodes_list))
    models = [i['class'] for i in instances]
    flow_lists = [[] for i in range(len(models))]
    for i in range(len(models)):
        for n in nodes_list:            
            flow_lists[i].append(flow[n][i])
    reg_model = rdt.RegularizedDecisionTreeClassifier(delta= 1., max_n_features=3, max_depth = 10, min_samples_leaf=1, min_samples_split=2)
    reg_model.fit(flow_lists, models)
    Then use as a regular Decision Tree Classifier
    '''
    def __init__(self, delta=1.0, max_n_features=3, max_depth = 3, min_samples_leaf = 1, min_samples_split = 2):
        self.delta = delta
        self.depth = 0
        self.max_n_features = max_n_features        
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split

    def nodeCounts(self,y):
        '''
        Calculates Classes counts in a given node
        '''        
        counts = []
        for one_class in range(len(self.classes_labels_)):
            c = y[y == one_class].shape[0]
            counts.append(c)
        return np.asarray(counts)

    def gini(self, y):
        '''
        Calculates gini criterion
        '''
        n_labels = len(y)
        if n_labels <= 1:
            return 0
        _,counts = np.unique(y, return_counts=True)
        probas = counts / n_labels       
        return 1 - np.sum(probas**2)

    def entropy(self, y):
        '''
        Calculates entropy criterion
        '''
        n_labels = len(y)
        if n_labels <= 1:
            return 0
        _,counts = np.unique(y, return_counts=True)
        probas = counts / n_labels
        n_classes = np.count_nonzero(probas)
        if n_classes <= 1:
            return 0

        ent = 0.
        # Compute entropy
        for i in probas:
            if i > 0.:
                ent -= i * np.log2(i)
        return ent
    
    def calcImpurity(self, y):
        '''
        Wrapper for the impurity calculation.
        '''
        # return self.gini(y)
        return self.entropy(y)
    
    def candidateFeatures(self, cols):
        '''
        Output list of features that can be used to compute best splits
        If max_n_features reached max_n_features then only output
        a list with already selected features + the default features (parameter of init())
        '''
        candidates = []  
        ## Cannot use any more features      
        if len(self.selected_features_) == self.max_n_features:
            candidates = list(self.selected_features_)
            ## add default features
            candidates.extend(self.default_features_)
        else:
            candidates = cols
        
        np.random.shuffle(np.array(candidates))
        return(candidates)
    
    def calcBestSplit(self, X, y):
        '''
        Calculates the best possible split for the current node of the tree
        '''        
        bestSplitCol = None
        bestThresh = None
        bestInfoGain = -999
        
        impurityBefore = self.calcImpurity(y)
        
        # List of possible feature for split
        candidates_features = self.candidateFeatures(range(X.shape[1]))

        # For each possible feature
        for col in candidates_features:
            x_col = X[:, col]
            ## if only single value
            # print(x_col)
            if np.min(x_col) == np.max(x_col):
                continue

            factor_reg = 1.
            # If the feature was never used before
            if col not in self.selected_features_:
                factor_reg = self.delta
            
            ## sort x_col
            # s_x_col = np.sort(x_col)
            s_x_col = np.unique(x_col)

            # for each sorted value in the column (expect the last)          
            for i in range(s_x_col.shape[0]-1):
                
                threshold = s_x_col[i]/2. + s_x_col[i+1]/2.

                y_right = y[x_col > threshold]
                y_left = y[x_col <= threshold]
                if y_right.shape[0] < self.min_samples_leaf  or y_left.shape[0]< self.min_samples_leaf:
                    continue

                # calculate impurity for the right and left nodes
                impurityRight = self.calcImpurity(y_right)
                impurityLeft = self.calcImpurity(y_left)

                # calculate information gain
                infoGain = impurityBefore
                infoGain -= impurityLeft * y_left.shape[0] / y.shape[0]
                infoGain -= impurityRight * y_right.shape[0] / y.shape[0]
                infoGain *= factor_reg               
                
                # is this infoGain better then all other?
                if infoGain >= bestInfoGain:
                    bestSplitCol = col
                    bestThresh = threshold
                    bestInfoGain = infoGain


        # if we still didn't find the split
        if bestInfoGain == -999:
            return None, None, None, None, None, None
        
        # making the best split        
        x_col = X[:, bestSplitCol]
        x_left, x_right = X[x_col <= bestThresh, :], X[x_col > bestThresh, :]
        y_left, y_right = y[x_col <= bestThresh], y[x_col > bestThresh]

        return bestSplitCol, bestThresh, x_left, y_left, x_right, y_right                
    
    def splitNode(self, X, y, node):
        # checking for the terminal conditions        
        if node.depth >= self.max_depth:
            node.is_terminal = True
            return  None, None, None, None

        if X.shape[0] < self.min_samples_split:
            node.is_terminal = True
            return  None, None, None, None

        if np.unique(y).shape[0] == 1:
            node.is_terminal = True
            return None, None, None, None

        # calculating current split
        splitCol, thresh, x_left, y_left, x_right, y_right = self.calcBestSplit(X, y)
                
        if splitCol is None:
            node.is_terminal = True
            return  None, None, None, None      

        # Do the child nodes have enough samples?
        if x_left.shape[0] < self.min_samples_leaf or x_right.shape[0] < self.min_samples_leaf:
            node.is_terminal = True
            return None, None, None, None  

        if splitCol not in self.default_features_:
            self.selected_features_.add(splitCol)      
        
        self.depth = max(self.depth,node.depth+1)
        node.column = splitCol
        node.threshold = thresh
        return x_left, y_left, x_right, y_right
    
    def fit(self, X, y, default_selected = []):
        '''
        Standard fit function to run all the model training
        Input
        -----
        X: Features
        y: classes to predicts
        prev_selected: list(int) indexes of variable  
        ''' 

        ## Get indexes of prev selected variables
        self.default_features_ = default_selected    

        ## Convert X,y into correct type
        X,y = check_X_y(X,y)
        self.n_features_in_ = X.shape[1]
 
        self.classes_labels_, self.classes_ = np.unique(y, return_inverse=True)
        self.selected_features_ = set()

        if True in np.iscomplex(y):
            raise ValueError('Complex data not supported')
        if X.shape[0]!= self.classes_.shape[0]:
            raise ValueError('X and y should have the same length')        

        # Root node creation
        self.tree_ = Node(0,self.nodeCounts(self.classes_),self.calcImpurity(self.classes_))

        ## Tree construction using BFS
        next_level = [(self.tree_, X, self.classes_)]
        while len(next_level) > 0:

            cur_level = next_level
            np.random.shuffle(cur_level)
            next_level = []

            for node, Xn, yn in cur_level:

                Xn_left, yn_left, Xn_right, yn_right = self.splitNode(Xn, yn, node)
                
                if not node.is_terminal:
                    node.left = Node(node.depth + 1,self.nodeCounts(yn_left),self.calcImpurity(yn_left))          
                    node.right = Node(node.depth + 1,self.nodeCounts(yn_right),self.calcImpurity(yn_right))

                    next_level.append((node.left, Xn_left, yn_left))
                    next_level.append((node.right, Xn_right, yn_right))

        return self
    
    def predictSample(self, x, node):
        '''
        Passes one object through decision tree and return the probability of it to belong to each class
        '''
        # if we have reached the terminal node of the tree
        if node.is_terminal:
            ## return probability vector
            c = node.counts
            return c / sum(c)
        
        if x[node.column] > node.threshold:
            probas = self.predictSample(x, node.right)
        else:
            probas = self.predictSample(x, node.left)            
        return probas        
    
    def predict(self, X):
        '''
        Returns the predicted labels for each X        
        X = np array
        '''     
        check_is_fitted(self)

        X =  check_array(X,dtype='numeric')
            
        predictions = []
        for x in X:
            pred = np.argmax(self.predictSample(x, self.tree_))
            predictions.append(self.classes_labels_[pred])
        
        return np.asarray(predictions)

    def predict_proba(self, X):
        '''
        Returns the probabilities for each X to belong to 
        each class        
        X = np array
        '''  
        check_is_fitted(self)

        X =  check_array(X,dtype='numeric')

        probas = []
        for x in X:
            p_x = self.predictSample(x, self.tree_)
            probas.append(p_x)
        return np.asarray(probas)


    def score(self, X, y):
        '''
        Return the average accuracy of the predictions for data X w.r.t. y
        '''

        check_is_fitted(self)

        X,y = check_X_y(X,y)

        n_good_predict = 0
        for i in range(X.shape[0]):
            pred = np.argmax(self.predictSample(X[i,:], self.tree_))
            if self.classes_labels_[pred] == y[i] :
                n_good_predict += 1
        return n_good_predict /  X.shape[0]
