import statistics

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import get_prototype as gp
from sklearn_lvq import glvq
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from scipy.spatial.distance import squareform, pdist
from sklvq.models import _glvq as glvqn
plt.style.use('ggplot')

max_glvq = 1000
max_glvqps = 1000
dataset1 = load_iris()
#dataset2 = load_wine()
#dataset3 = load_breast_cancer()
datasets = [
    dataset1
]
figure = plt.figure(figsize=(7, 5))
i = 1
for ds_cnt, dataset in enumerate(datasets):
    X, y = dataset.data, dataset.target
    plt.cla()
    pca = PCA(n_components=2, random_state=42)
    pca.fit (X)
    X = pca.transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42, test_size=0.3)

    plot_step = 0.04

    # define bounds of the domain
    min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
    min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1

    # define the x and y scale
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)

    # create all the lines and rows of the grid
    xx, yy = np.meshgrid(np.arange(min1, max1, plot_step),
                         np.arange(min2, max2, plot_step))

    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1,r2))
    ax = plt.subplot(len(datasets), 2 + 1, i)
    if ds_cnt == 0:
        ax.set_title ("Input data")
    # Plot the training points
    ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.Spectral, edgecolors="k", s=50)
    # Plot the testing points

    ax.scatter(
    x_test[:, 0], x_test[:, 1], c=y_test, cmap=plt.cm.Spectral, alpha=0.6, edgecolors="k", s=50
    )
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    ax = plt.subplot(len(datasets), 2 + 1, i)
    # define the model
    proto = gp.getpro (X, y)
    ps_prototypes = proto.prototypes
    ps_prototype_label = proto.prototype_label
    zip_proto = np.column_stack ((ps_prototypes, ps_prototype_label))
    model = glvq.GlvqModel (prototypes_per_class=proto.prototype_number, initial_prototypes=zip_proto,
                            max_iter=max_glvqps)

    # fit the model
    model.fit (x_train, y_train)
    # make predictions for the grid
    ps_yhat = model.predict (grid)
    ps_ypred = model.predict (x_test)
    acc_score_ps = accuracy_score (ps_ypred, y_test)
    cl_ps = classification_report (ps_ypred, y_test)
    print (cl_ps)
    yhat = np.array (ps_yhat)
    # reshape the predictions back into a grid

    zz = yhat.reshape(xx.shape)

    # plot the grid of x, y and z values as a surface
    plt.contourf(xx, yy, zz, cmap=plt.cm.Spectral)
    glvqpspro = model.w_
    # create scatter plot for samples from each class
    for class_value in range(2):
        # get row indexes for samples with this class
        row_ix = np.where(y == class_value)
        # create scatter of these samples
        # Plot the training points
        ax.scatter (x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.Spectral, edgecolors="k", s=50)
        # Plot the testing points

        ax.scatter (
        x_test[:, 0], x_test[:, 1], c=y_test, cmap=plt.cm.Spectral, alpha=0.3, edgecolors="k",
        s=50)
        ax.scatter (glvqpspro[:, 0], glvqpspro[:, 1], c=model.c_w_, cmap=plt.cm.Spectral, marker='X',
                    edgecolors="k", s=200)
    ax.text (
            xx.max () - 0.3,
            yy.min () + 0.3,
            ("%.2f" % acc_score_ps ).lstrip ("0"),
            size=15,
            horizontalalignment="right",
        )
    ax.axes.xaxis.set_visible (False)
    ax.axes.yaxis.set_visible (False)
    if i == 2:
        plt.title("GLVQ-PS")
    i += 1
######################Generalised Learning Vector Quantisation###########################
    glvq1 = glvq.GlvqModel (prototypes_per_class=proto.prototype_number, max_iter=max_glvq)
    lvq_methods = [glvq1]
    lvq_methods1 = ['GLVQ']

    print('Number of Prototype in GLVQ is ', len(proto.prototypes))
    ax = plt.subplot(len(datasets), 2 + 1, i)
    # fit the model
    glvq1.fit(x_train, y_train)

    # make predictions for the grid
    yhat1 = glvq1.predict(grid)
    ypred1 = glvq1.predict(x_test)
    test_accuracy =  accuracy_score(y_test, ypred1)
    cl = classification_report(ypred1, y_test)
    print(cl)
    # reshape the predictions back into a grid
    zz = yhat1.reshape(xx.shape)

    # plot the grid of x, y and z values as a surface
    plt.contourf(xx, yy, zz, cmap=plt.cm.Spectral)

    # create scatter plot for samples from each class
    prototypes_glvq = glvq1.w_
    for class_value in range(2):
        # get row indexes for samples with this class
        row_ix = np.where(y == class_value)
        # create scatter of these samples
        # Plot the training points
        ax.scatter (x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.Spectral, edgecolors="k", s=50)
        # Plot the testing points

        ax.scatter (
        x_test[:, 0], x_test[:, 1], c=y_test, cmap=plt.cm.Spectral, alpha=0.3, edgecolors="k", s=50
        )
        ax.scatter (prototypes_glvq[:, 0], prototypes_glvq[:, 1], c=glvq1.c_w_, cmap=plt.cm.Spectral, marker='X', edgecolors="k", s=200 )
        ax.text(
            xx.max () - 0.3,
            yy.min () + 0.3,
            ("%.2f" % test_accuracy).lstrip ("0"),
            size=15,
            horizontalalignment="right",
        )
    if i == 3:
        plt.title ("GLVQ")
    ax.axes.xaxis.set_visible (False)
    ax.axes.yaxis.set_visible (False)
    i += 1
plt.tight_layout()
plt.savefig('Iris.png', dpi=300)
plt.gca()
plt.show()
#####################################EXIT################################################