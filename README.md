# Prototype Selection for Interpretable Classification
Given a training data set $V=\{v_1,\cdots,v_n\} \in \mathbb{R}^m$ of n data samples, where each sample is representing an m-dimensional feature vector. Each sample in the training set is associated with corresponding class label $y_1,\cdots,y_n \in \{1,\cdots, L\}$. PS scheme returns a set $\mathscr{P}_l$ for each class $l \in \{1,\cdots, L\}$. The returned output set is a condensed prototype set $\mathscr{P} =\{\mathscr{P}_1,\cdots,\mathscr{P}_L\} \subseteq V$ that can be seen as having an interpretable meaning. The prototype selection in PS is based on three important notion :
- Prototypes should cover as many training samples of the same class $l$
- Prototypes should cover few training samples from another class and,
- The number of prototypes needed to cover data samples of a prticular class should be as few as possible (also called sparsity).

The goal of the interpretaility is to bring trust, privacy, fairness and robustness to the Machine Learning models in comparison with decision of black-box models that are hard to comprehend.
