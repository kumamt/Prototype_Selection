# Interpretability
Most ML algorithms require the ability to explain the decision taken by a model easily. For example, in banks, when a person requests a bank loan. The bank should be able to explain why a particular decision (rejection or acceptance) is made. The goal of the interpretaility is to bring **trust**, **privacy**, **fairness** and **robustness** to the Machine Learning models in comparison with decision of black-box models that are hard to comprehend.

# Prototype Selection for Interpretable Classification (PS)
Given a training data set $V=\{v_1,\cdots,v_n\} \in \mathbb{R}^m$ of n data samples, where each sample is represents an m-dimensional feature vector. Each sample in the training set is associated with corresponding class label $y_1,\cdots,y_n \in \{1,\cdots, L\}$. PS scheme returns a set $\mathscr{P}_l$ for each class $l \in \{1,\cdots, L\}$. The returned output set is a condensed prototype set $\mathscr{P} =\{\mathscr{P}_1,\cdots,\mathscr{P}_L\} \subseteq V$ that can be seen as having an interpretable meaning. According to _J. Bien and R. Tibshirani_  the prototype selection in PS is based on three important notion :
- Prototypes should cover as many training samples of the same class $l$
- Prototypes should cover few training samples from another class and,
- The number of prototypes needed to cover data samples of a particular class should be as few as possible (also called sparsity).

The prototypes selected in PS are actual data points as they will add more interpretable meaning to the model. PS scheme intially is formed using Set Cover Integer problem. For a given radius of an epsilon ball (centered at chosen prototype) PS outputs minimum number of balls required to form a Cover while preserving the properties of prototypes. Then, is tranformed into _l_-prize collection problem and solving using two approach namely
- Greedy Approach (recommended for large dataset)
- Randommized Rounding Algorithm

For further reading and mathematical understanding please refer [J. Bien and R. Tibshirani](https://arxiv.org/pdf/1202.5933.pdf).

# Generalised Learning Vector Quantisation (GLVQ)
In order to understand the GLVQ, the prototype set has been considered as  $W = \{w_1,\cdots,w_l\}$ for each class in the dataset where, $l\in \{1,\cdots,L\}$ and $L$ is the set of class labels. For comparison of samples among the prototypes, a distance matrix has been represented by $**D**_{(i,j)} = d(v_i,w_j)$ where $d$ is a differentiable dissimilarity measure. Next to this, a classifier (relative distance difference) function for GLVQ is defined as,

!<img src="https://github.com/amitk0693/Prototype_Selection/blob/38ebb6b86a97a01ae6943c9210dd0c1452bed4c2/GLVQ.png" width="500" height="100">

where, $w^+$ is the best matching correct prototype with $c(w^+)=c(v)$ and $w^-$ is the closest prototype belonging to the wrong or incorrect class $c(w^+)\neq c(v)$. For correct classification the distance $d(v, w^+)$ of the data point belonging to the correct class prototype should be smaller than the distance $d(v, w^-)$ of the prototype belonging to the incorrect class. In this case, the output of the classifier function are negative values, and hence, the cost function of $E_{GLVQ}$ is then an approximation of the overall classification error. 

For further reading and udestanding please refer [A. Sato and K. Yamada](https://proceedings.neurips.cc/paper/1995/file/9c3b1830513cc3b8fc4b76635d32e692-Paper.pdf)

# Experiments:
## Iris data 
Image below shows the selected prototype (**X**) to represent the data samples (filled **o**).
- Unlike k-Nearest Neighbor storing whole data for prediction, in prototype selection scheme the condensed form of training data samples (prototypes) are only require to be stored saving large amount of memory.
- For prediction it only utlises the distances to the selected prototypes (saving time required to compare whole data sample)
!<img src="https://github.com/amitk0693/Prototype_Selection/blob/2139abc148490df75f42f6aacfa5a602116b2cf0/Iris.png" width="50" height="50">




