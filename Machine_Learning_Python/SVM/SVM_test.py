############################### SVM FROM SCRATCH ###########################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style


style.use('ggplot')


############################################################################################
class Support_Vector_Machine:
    '''
    CONSTRUCTOR METHOD INICIALIZE THE OBJECT
    '''

    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {-1: 'b', 1: 'r'}

        ## FIRST CHECK
        if visualization:
            self.figure = plt.figure()
            self.ax = self.figure.add_subplot(1, 1, 1)  ##EQ TO SUBPLOT ON MATLAB

    ## METHODS
    '''
    THIS METHOD COMPUTE THE HYPERPLANES GIVEN A DATASET. THIS METHOD TAKES DATASET IN DICTIONARY FORMAT
    THIS METHOD CALCULATE THE VECTOR W AND THE BIAS b
    THE HYPERPLANE EQUATIONS IS GIVEM BY W*X+b
    '''

    def train(self, data):
        self.data = data
        ## DICTIONARY WITH THE MAGNITUDES OF THE 'FIT' VECTORES
        opt_dict = {}

        transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]

        all_data = []

        for yi in self.data:
            for featureset in self.data[yi]:
                for features in featureset:
                    all_data.append(features)

        self.max_features_val = max(all_data)
        self.min_features_val = min(all_data)

        all_data = None

        step_sizes = [self.max_features_val * 0.1, self.max_features_val * 0.01,
                      self.max_features_val * 0.01]  ## ABOVE THIS POINT IS COMPUTATIONAL EXPENSIVE

        ## COMPUTATIONAL EXPENSIVE
        b_range_multiple = 5

        ##
        b_multiple = 5

        ##
        latest_optimum = self.max_features_val * 10

        ##
        for step in step_sizes:
            W = np.array([latest_optimum, latest_optimum])
            optimized = False  ## CONVEX PROBLEM
            while not optimized:
                for b in np.arange(-1 * (self.max_features_val * b_range_multiple),
                                   self.max_features_val * b_range_multiple, step * b_multiple):
                    for transformation in transforms:
                        w_t = W * transformation
                        found_option = True
                        ##
                        ##
                        ##
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False
                        ##
                        ##
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                ##
                ##
                if W[0] < 0:
                    optimized = True
                    print('OPTIMIZED A STEP...')
                else:
                    W = W - step
            ##
            ##
            norms = sorted(n for n in opt_dict)

            ##
            ##
            opt_choice = opt_dict[norms[0]]
            self.W = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2

    '''
    THIS METHOD USE THE SVM TRAINED AND PREDICT THE NEW DATA ABSE ON THE HYPERPLANES DEFINED
    '''

    def predict(self, features):

        ## CHECKING THE SIGN
        classification = np.sign(np.dot(np.array(features), self.W) + self.b)

        ##
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], marker='*', s=200, c=self.colors[classification])

        ##
        return classification

    '''
    THIS METHOD IS USE TO VISUALIZE THE DATA CLASSIFID BY THE SVM PREDICT METHOD
    '''

    def visualize(self):
        ##
        [[self.ax.scatter(x[0], x[1], s=100, c=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        ##
        def hyperplane(x, w, b, v):
            ##
            return (-w[0] * x + v - b) / w[1]

        ## LIMITES FOR THE PLOT
        datarange = (self.min_features_val * 0.9, self.max_features_val * 1.1)
        hyper_x_min = datarange[0]
        hyper_x_max = datarange[1]

        ## POSITIVE SUPORT HYPERPLANE (PSH)
        ## (W*x+b)=1
        psh_1 = hyperplane(hyper_x_min, self.W, self.b, 1)
        psh_2 = hyperplane(hyper_x_max, self.W, self.b, 1)

        ## PLOTING THE HYPERPLANE
        self.ax.plot([hyper_x_min, hyper_x_max], [psh_1, psh_2], 'k')

        ## NEGATIVE SUPORT HYPERPLANE (NSH)
        ## (W*x+b)=-1
        nsh_1 = hyperplane(hyper_x_min, self.W, self.b, -1)
        nsh_2 = hyperplane(hyper_x_max, self.W, self.b, -1)

        ## PLOTING THE HYPERPLANE
        self.ax.plot([hyper_x_min, hyper_x_max], [nsh_1, nsh_2], 'k')

        ## DECISION BONDARY HYPERPLANE (DBH)
        ## (W*x+b)=1
        dbh_1 = hyperplane(hyper_x_min, self.W, self.b, 0)
        dbh_2 = hyperplane(hyper_x_max, self.W, self.b, 0)

        ## PLOTING THE HYPERPLANE
        self.ax.plot([hyper_x_min, hyper_x_max], [dbh_1, dbh_2], 'y--')

        ## PLOTING EVERYTHING
        plt.show()


## END OF THE CLASS
############################################################################################

## TEST
data_dict = {-1: np.array([[1, 7], [2, 8], [3, 8], ]), 1: np.array([[5, 1], [6, -1], [7, 3], ])}

## EXAMPLE
hypo = Support_Vector_Machine()
hypo.train(data_dict)

## PREDICTION TEST
predict_us = [[0, 10], [1, 3], [3, 4], [3, 5], [5, 5], [5, 6], [6, -5], [5, 8], [3, 2], [1, 1], [2, 2]]

for i in predict_us:
    hypo.predict(i)

hypo.visualize()