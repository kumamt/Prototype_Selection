import cvxpy as cvx
import numpy as np
from scipy.stats import bernoulli
from cvxpy import GUROBI as solverGUROBI

class prototype_classifier:

    """ A ball is drawn around every element in the distance matrix with radius
        epsilon. The element whose ball covers most other elements is selected
        as prototype. All such covered elements and the new prototype are
        removed for the next round. This is repeated until no balls cover more
        than its center element.
    	
        Parameters
        ----------
        _epsilon: float
            A hyperparameter, indicating the region covered by the prototypes with radius epsilon.

        Attributes
        ----------
        self.X: M x N matrix
            Training data set.
        self.y: M x 1 vector
            Class label of the training data set self.X.
        self.lengthX: int
            Training data length.
        self._epsilon: float
            Ball radius.
        self._lambda: float
            a scalar for normalisation.
    """

    def __init__(self, _epsilon):

            self._lambda = None
            self.c_w_ = None
            self.w_ = None
            self.X = None
            self.y = None
            self.lengthX = None
            self._epsilon = _epsilon
            self.lengthL = None
            self.label = None
            self.data_info = None
            # Variables for solving relaxed Integer Program.
            self.xi_n_lin = None
            self.alpha_j_lin = None
            self.opt_val = None
            # Variables for solving the Randomized Rounding Algorithm.
            self.prototype_length = None
            self.random_round_Sn = None
            self.random_round_Aj = None
            self.random_round_optimal_val = None

    def labelsInfo(self):
        """ Extracts the unique class label from the data X.

            Attributes
            ----------
            self.label : set 
                Stores unique class labels.

            Returns
            -------
            set of class labels.
        """
        self.label = set(self.y)
        return self.label

    def getNumLabel(self):
        """ Extracts the length of the class labels.

            Attributes
            ----------
            self.lengthL : int
                length of the class labels.
            
            Returns
            -------
            self.length : int
                length of the class labels.
           
        """
        self.lengthL = len(self.label)
        return self.lengthL 

    def dataInfo(self, label):
        """Creates a data dictionary according to the class label passed as argument label.

            Parameter
            ---------
            label: int
                contains the label number.

            Attributes
            ----------
            indexes: list[int]
                indices, where label is equal to the class label of the data.
            X_l: data
                subset of the training data with indexes as index.
            y_l: data
                class label of the data X_l.
            size_l: int
                length of the data with the same label.
            indexes_not_l: list[int]
                indices, where label is not equal to the class label of the data.
            X_not_l: data
                subset of data where the label is not equal to the class label.
            y_not_l: data
                contains the label which is not equal to the label.
            size_not_l: list[int]
                length of the data which is not of the same label.
            data: dict
                contains the data where argument label is equal to the class label.
            
            Returns
            -------
            data: data dictionary containing data with class label equal to label and class label not
                equal to label.
        """
        indexes = [idx for idx, lbl in enumerate(self.y) if lbl == label]
        X_l = self.X[indexes]
        y_l = self.y[indexes]
        size_l = X_l.shape[0]
        indexes_not_l = [idx for idx, lbl in enumerate(self.y) if lbl != label]
        X_not_l = self.X[indexes_not_l]
        y_not_l = self.y[indexes_not_l]
        size_not_l = X_not_l.shape[0]
        data = {
            "X_l": X_l,
            "y_l": y_l,
            "indices_l": indexes,
            "size_l": size_l,
            "X_not_l": X_not_l,
            "y_not_l": y_not_l,
            "indices_not_l": indexes_not_l,
            "size_not_l": size_not_l,
        }
        return data

    def checkNeighborhood(self, x, x_test):
        """ Checks if a point is in the epsilon ball or if a point 'x_test' is in the epsilon neighborhood of a
            point 'x'.
        
            Parameter
            ---------
            x: vector
                center of the epsilon ball.
            x_test: vector
                point to be checked if it lies in the ball with radius epsilon or not.
            
            Attribute
            ---------
                result: bool
                    Checks if the point is in neighbourhood or not.
            
            Returns
            -------
                True, if a point lies in the neighborhood of epsilon ball centered around parameter
                x otherwise, returns False.
        """
        result = np.linalg.norm((x - x_test), ord=2, keepdims=True) <= self._epsilon
        return result

    def calculate_Clj(self, label):
        """ Calculates the total number of data samples(with different label) if it comes in the epsilon 
            neighborhood of a point if a point is chosen as a prototype.

            Parameter 
            ---------
            label: int
                contains a class label.
            
            Attributes
            ----------
            data: dict
                contains the data dictionary according to the argument label.
            X_l: Any
                contains the data with same class label.
            X_not_l: Any
                contains the data where label is not equal to class label.
            sets_C_lj: list[int]
                contains number of points covered where, class label is not equal to true label if x_jl is
                considered as prototype.
            
            Returns
            -------
            Cost of adding a point.
        """
        data = self.dataInfo(label=label)
        X_l = data["X_l"]
        X_not_l = data["X_not_l"]
        sets_C_lj = []
        for x_jl in X_l:
            temp = 0
            for x_no_l in X_not_l:
                if self.checkNeighborhood(x_jl, x_no_l):
                    temp += 1
            sets_C_lj.append(temp)
        C_lj = [(self._lambda + set_C_lj) for set_C_lj in sets_C_lj]
        data["C_lj"] = np.array(C_lj)
        return data

    def checkCoverPoints(self, label):
        """ Checks for neighborhood points, if a point is in the ball with radius epsilon(for the
            chosen data as a prototype) then, the value for that index is 1 else it is zero.

            Parameter
            ---------
            label: int
                Contains a class label.

            Attributes
            ----------
            X_l: data
                subset of the training data with indexes as index.
            size_l: int
                length of the data with the same label.
            constraint_matrix: int
                size_l X size_l matrix which contains the neighborhood information.

            Returns
            -------
            data:  updates the data dictionary with pairwise neighborhood information.
        """
        data = self.calculate_Clj(label=label)
        X_l = data["X_l"]
        size_l = data["size_l"]
        constraint_matrix = np.zeros(shape=(size_l, size_l))
        for x_nl in range(size_l):
            for x_jl in range(size_l):
                if self.checkNeighborhood(X_l[x_jl], X_l[x_nl]):
                    constraint_matrix[x_nl][x_jl] = 1
        data["constraint_matrix"] = constraint_matrix
        return data

    def fit(self, train_X, train_y,):
        """ Here the model is trained according to the class label using convex optimisation method.
            Attributes are added to the data dictionary for Randomized rounding algorithm for optimal
            parameters. 

            Attributes
            ----------
            train_X: M x N matrix
                Training data set.
            train_y: M x 1 vector
                Class label of the training data set.
            label: int
                contains set of unique labels.
            alpha_j_lin: vector
                stores the alpha value after optimisation.
            xi_n_lin: vector
                stores Xi value after optimisation.
            opt_val_lin: list
                contains the optimal value of the Objective function.
            update_data_info:
                updates the data dictionary with the attributes above.
        """
        self.X = train_X
        self.y = train_y
        self.lengthX = len(self.X)
        self._lambda = 1/ self.lengthX
        label = self.labelsInfo()
        alpha_j_lin = np.zeros(shape=self.lengthX)
        xi_n_lin = np.zeros(shape=self.lengthX)
        opt_val_lin = []
        update_data_info = []

        for lbl in label:
            data = self.checkCoverPoints(label=lbl)
            size_l = data["size_l"]
            C_lj = data["C_lj"]
            alpha_jl = cvx.Variable(shape=size_l)
            xi_nl = cvx.Variable(shape=size_l)
            constraint_matrix = data["constraint_matrix"]
            zero_vec = np.zeros(shape=size_l)
            one_vec = np.ones(shape=size_l)
            # Objective function for minimisation.
            objective = cvx.Minimize((C_lj @ alpha_jl) + sum(xi_nl))
            # Constraints for the minimisation.
            constraints = [(constraint_matrix @ alpha_jl >= (one_vec - xi_nl)),
                           alpha_jl >= zero_vec,
                           alpha_jl <= one_vec,
                           xi_nl >= zero_vec]

            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=solverGUROBI)

            data["alpha_jl_lp"] = [1 if alpha_jl >= 1 else alpha_jl for alpha_jl in alpha_jl.value]
            data["alpha_jl_lp"] = [0 if alpha_jl <= 0 else alpha_jl for alpha_jl in data["alpha_jl_lp"]]
            data["xi_nl_lp"] = [1 if xi_nl >= 1 else xi_nl for xi_nl in xi_nl.value]
            data["xi_nl_lp"] = [0 if xi_nl <= 0 else xi_nl for xi_nl in data["xi_nl_lp"]]

            first_term = sum([(c_lj * alpha_jl) for c_lj, alpha_jl in zip(data["C_lj"], data["alpha_jl_lp"])])
            second_term = sum(data["xi_nl_lp"])
            # calculate the optimal value after the optimisation.
            optimal_l = first_term + second_term

            data["optimal_l_lp"] = optimal_l
            opt_val_lin.append(optimal_l)
            update_data_info.append(data)

            indices_l = data["indices_l"]
            for idx in indices_l:
                alpha_j_lin[idx] = alpha_jl.value[indices_l.index(idx)]
                xi_n_lin[idx] = xi_nl.value[indices_l.index(idx)]

        self.alpha_j_lin = alpha_j_lin
        self.xi_n_lin = xi_n_lin
        self.opt_val = sum(opt_val_lin)
        self.data_info = update_data_info
        self.objectiveValue()

    def objectiveValue(self):
        """ Here Randomized Rounding Algorithm is performed to recover the integer values we need 
            for our indicator variable, we choose to round our optimal variables of the linear program.

            Attributes
            ----------
            data_info:

            label:
                 contains set of unique labels.
            random_round_Aj:

            random_round_Sn:
            random_round_optimal_value:
            update_data_info:
        """
        data_info = self.data_info
        label = self.labelsInfo()
        random_round_Aj = np.zeros(shape=self.lengthX)
        random_round_Sn = np.zeros(shape=self.lengthX)
        random_round_optimal_value = []
        update_data_info = []
        for lbl in label:
            data = data_info[lbl]
            size_l = data["size_l"]
            temp_random_round_Ajl = np.zeros(shape=size_l, dtype=int)
            temp_random_round_Snl = np.zeros(shape=size_l, dtype=int)
            temp_optimal_round = float('nan')
            C_lj = data["C_lj"]
            alpha_jl_lp = data["alpha_jl_lp"]
            xi_nl_lp = data["xi_nl_lp"]
            optimal_l_lp = data["optimal_l_lp"]
            optimal_l_log = 2 * np.log (size_l) * optimal_l_lp
            repeat = True
            while repeat:
                temp_random_round_Ajl = np.zeros(shape=size_l, dtype=int)
                temp_random_round_Snl = np.zeros(shape=size_l, dtype=int)

                for l in range(int(np.ceil(2 * np.log(size_l)))):
                    for j in range(size_l):
                        temp_round_Ajl = bernoulli.rvs(alpha_jl_lp[j])
                        temp_random_round_Ajl[j] = max(temp_round_Ajl, temp_random_round_Ajl[j])
                        temp_round_Snl = bernoulli.rvs(xi_nl_lp[j])
                        temp_random_round_Snl[j] = max(temp_round_Snl, temp_random_round_Snl[j])

                constraint_matrix = data["constraint_matrix"]
                first_term = sum([c_lj * A_jl for c_lj, A_jl in zip(C_lj, temp_random_round_Ajl)])
                second_term = sum(temp_random_round_Snl)
                temp_optimal_round = first_term + second_term

                get_val_Ajl = constraint_matrix @ temp_random_round_Ajl
                get_val_Sn = 1 - temp_random_round_Snl
                if all ([lhs >= rhs for lhs, rhs in zip(get_val_Ajl, get_val_Sn)]):
                    if all ([(A_jl == 0 or A_jl == 1) for A_jl in temp_random_round_Ajl]):
                        if all ([S_nl >= 0 for S_nl in temp_random_round_Snl]):
                            if temp_optimal_round <= optimal_l_log:
                                repeat = False

            data["random_round_Ajl"] = temp_random_round_Ajl
            data["random_round_Snl"] = temp_random_round_Snl
            data["optimal_val_random"] = temp_optimal_round
            random_round_optimal_value.append(temp_optimal_round)
            update_data_info.append(data)

            indices_l = data["indices_l"]
            for idx in indices_l:
                random_round_Aj[idx] = temp_random_round_Ajl[indices_l.index(idx)]
                random_round_Sn[idx] = temp_random_round_Snl[indices_l.index(idx)]

        self.random_round_Aj = random_round_Aj
        self.random_round_Sn = random_round_Sn
        self.random_round_optimal_val = sum(random_round_optimal_value)

        self.w_ = [x for x, A_j in zip(self.X, self.random_round_Aj) if A_j == 1]
        self.w_ = np.array (self.w_)

        self.c_w_ = [y for y, A_j in zip(self.y, self.random_round_Aj) if A_j == 1]
        self.prototype_length = self.w_.shape[0]
        self.data_info = update_data_info

    def predict(self, x_predict):
        """ Test Error: After choosing prototypes, classify a point as the label of the nearest 
            prototype.
            Cover Error: After choosing prototypes, classify a point to the class label of the nearest
            prototype if it lies in epsilon range of the prototype, If no such prototypes
            exist then count it as mis-classification.

            Parameter
            ---------
            x_predict:
                unseen data to test the model.
            
            Attributes
            ----------
            prototypes: list | ndarray
                contains chosen prototypes.
            prototype_label: list
                contains the label of the prototypes.
            y_test_predict:  list
                test error for mis-classification.
            y_cover_predict: list
                cover error for mis-classification.
            
            Returns
            -------
            y_test_predict: list
            y_cover_predict: list
        """


        prototypes = self.w_
        prototype_label = self.c_w_

        y_test_predict = []
        y_cover_predict = []

        for i in x_predict:
            distances = []
            neighborhood = []
            for x in prototypes:
                dist = np.linalg.norm((x - i), ord=2, keepdims=True)
                distances.append(dist)
                neighborhood.append(dist <= self._epsilon)

            idx = np.argmin(distances)
            y_test_predict.append(prototype_label[idx])

            if sum(neighborhood) == 1.0:
                idx = neighborhood.index(1)
                y_cover_predict.append(prototype_label[idx])

            if sum(neighborhood) > 1.0:
                labels = [label for label, y_index in zip(prototype_label, neighborhood) if y_index == 1]
                if labels.count(labels[0]) == len(labels):
                    y_cover_predict.append(labels[0])
                else:
                    y_cover_predict.append(-2)

            if sum(neighborhood) < 1.0:
                y_cover_predict.append(-50)

        return y_test_predict