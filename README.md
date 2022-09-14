# Interpretability
Most ML algorithms require the ability to explain the decision taken by a model easily. For example, in banks, when a person requests a bank loan. The bank should be able to explain why a particular decision (rejection or acceptance) is made. The goal of the interpretaility is to bring **trust, privacy, fairness** and **robustness** to the Machine Learning models in comparison with decision of black-box models that are hard to comprehend.
# Prototype Selection for Interpretable Classification
Given a training data set $V=\{v_1,\cdots,v_n\} \in \mathbb{R}^m$ of n data samples, where each sample is representing an m-dimensional feature vector. Each sample in the training set is associated with corresponding class label $y_1,\cdots,y_n \in \{1,\cdots, L\}$. PS scheme returns a set $\mathscr{P}_l$ for each class $l \in \{1,\cdots, L\}$. The returned output set is a condensed prototype set $\mathscr{P} =\{\mathscr{P}_1,\cdots,\mathscr{P}_L\} \subseteq V$ that can be seen as having an interpretable meaning. According to [J. Bien and R. Tibshirani](https://arxiv.org/pdf/1202.5933.pdf) the prototype selection in PS is based on three important notion :
- Prototypes should cover as many training samples of the same class $l$
- Prototypes should cover few training samples from another class and,
- The number of prototypes needed to cover data samples of a particular class should be as few as possible (also called sparsity).

# Generalised Learning Vector Quantisation 
  

# Experiments performed :
- Iris data 
