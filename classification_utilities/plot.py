import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn as sns

from sklearn.metrics import accuracy_score 
from sklearn.metrics import auc 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve


def plot_confusion_matrix(matrix):
    """
    Plot a confusion matrix by adding annotation
    
    Parameters
    ----------
    
    matrix: numpy.ndarray. Confusion matrix
    """
    
    df_cm = pandas.DataFrame(matrix, range(2),range(2))
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, annot=True,annot_kws={"size": 16})
    plt.show()

def compute_and_print_scores(X1, X2, y1, y2, model, labels, threshold=None):
    """
    Utility to monitor the performance of a classifier on train and test set.
    Returns confusion matrix and different classification scores 
    for a classifier, on both train and test set.
    
    Parameters
    ----------
    
    X1: numpy.ndarray. Training input data.
    X2: numpy.ndarray. Test input data.
    y1: numpy.ndarray. Training output target.
    y2: numpy.ndarray. Test output target.
    model: sklearn learner object. Trained model.
    labels: list. List of labels to assign at each dataset
    threshold: float. Value to be used to set the binary classification boundary, optional.
    
    Returns
    -------
    
    Summary of performance
    """
    
    
    for label_set,X,y in zip(labels,[X1, X2], [y1, y2]):
        if threshold is not None:
            prediction = model.predict_proba(X)[:,1] > threshold
        else:
            prediction = model.predict(X)
        print(label_set+' set:')
        for label_score, scorer in zip(['precision', 'recall', 'F1', 'accuracy'],[precision_score, recall_score, f1_score, accuracy_score]):
            score = scorer(y, prediction)
            message = '{} score = {}'.format(label_score, score)
            print(message)
        matrix=confusion_matrix(y, model.predict(X))
        plot_confusion_matrix(matrix)

def find_best_classification_threshold(X_train, y_train,
                                      classifier):
    """
    Returns the plot of a ROC curve for a given classifier, 
    along with an estimate of AUC and the optimal decision threshold.
    
    Parameters
    ----------
    
    X_train: numpy.ndarray. Training input data.
    y_train: numpy.ndarray. Training output target.
    classifier: sklearn learner object. Trained model.
    
    Returns
    -------
    
    ROC curve plot
    """

    y_score = classifier.predict_proba(X_train)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[0], tpr[0], thresholds = roc_curve(y_train, y_score[:, 1])
    roc_auc[0] = auc(fpr[0], tpr[0])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_train.ravel(), y_score[:,1].ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    optimal_idx = numpy.argmax(tpr[0] - fpr[0])
    optimal_threshold = thresholds[optimal_idx]
    
    plt.figure(figsize=(12,8))
    plt.plot(fpr[0], tpr[0], lw=2, label='ROC curve (area = %0.2f}' % roc_auc[0])  
    plt.plot(fpr[0][optimal_idx], tpr[0][optimal_idx], lw=2, color='r', marker='o', 
             label='Optimal threshold = {}'.format(optimal_threshold))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend()
    plt.show()
