import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import seaborn as sns
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import  classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from inputfeatures import attributes
from target import amount
import io
import base64

def create_io_obj():
    """ This function creates an image object that

        Returns   
        ----------
        graph_url: base64
            image object
    """
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    graph_url = base64.b64encode(bytes_image.getvalue()).decode()
    plt.close()

    return 'data:image/png;base64,{}'.format(graph_url)



def plot_confusion_matrix(y_true, y_pred, normalize=False, 
                          title=None, cmap=plt.cm.Blues, figsize=(10, 10)):
    """ This function plots the confusion matrix.
        Parameters
        ----------
        y_true: array
            true labels
        y_pred: array
            predicted labels   
        normalize: bool
            whether to normalise the matrix
        title: str
            custom title (optional)
        cmap: matplotlib colormap
            colormap to use for matrix
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=20)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes)
    # Set alignment and fontsize of title, labels, tick labels
    plt.setp(ax.set_title(label=title), fontsize=20)
    plt.setp(ax.set_xlabel(xlabel='True label'), fontsize=20)
    plt.setp(ax.set_ylabel(ylabel='Predicted label'), fontsize=20)
    plt.setp(ax.get_xticklabels(), fontsize=20, rotation=0, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), fontsize=20)
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center", fontsize=20,
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



def  RandomForest(db: str, cf_matrix: bool):
    """ This function tests the performance of a Random Forest Classifier on
        extracted features and target variable from sqlite database

        Parameters
        ----------
        db: str
            name of sqlite databases
        cf_matrix: bool
            whether to display the confusion matrix
    """

    conn = sqlite3.connect(db)

    df = pd.read_sql_query('Select  *  From  EmployeesTransactions', conn)
    
    # Exclude Null Income values for better performance
    df =  df[df['income_null'] != 1]

    X =  df.drop(columns=['emp_id', 'amount'])
    y = df.loc[:, 'amount']

    # Normalizing Features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    if cf_matrix == False:
        
        class_labels = list(range(len(df['amount'].unique())))
        y = label_binarize(y, classes=class_labels)

        if len(class_labels) == 2:
            y = np.ravel(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        # Grid Search

        param_grid  = {'n_estimators':[10, 20, 30], 'max_depth':[2, 3, 4, 5], 'min_samples_split':[5, 10, 15]}
        # param_grid  = {'n_estimators':[10], 'max_depth':[2], 'min_samples_split':[5]}

        rf_model = RandomForestClassifier(random_state=0)
        grid  =  GridSearchCV(rf_model, param_grid,  cv=5,  refit=True, verbose=2)
        grid.fit(X_train, y_train)

        
        # One Vs Rest Classification : Multi-Class Classification

        rf_model = OneVsRestClassifier(RandomForestClassifier(n_estimators=grid.best_params_['n_estimators'],\
                max_depth=grid.best_params_['max_depth'],\
                min_samples_split=grid.best_params_['min_samples_split'],\
                random_state=0)).fit(X_train, y_train)
        

        if len(class_labels) > 2:
            N = len(class_labels)
            rf_score = rf_model.predict_proba(X_test)
        else:
            N  = 1
            rf_score = rf_model.predict_proba(X_test)[:, 1].reshape(-1, 1)
        
        y_test = y_test.reshape(-1, N)
        rf_predictions = rf_model.predict(X_test).reshape(-1, N)

        # Accuracy, Precision. Recall, F1-Score, and ROC Curves
        
        accuracy = dict()
        precision, recall = dict(), dict()
        f1score = dict()
        fpr, tpr = dict(), dict()
        roc_auc = dict()
        bytes_image1 = dict()        
        bytes_image2 = dict()

        for i in range(0, N):

            accuracy[i] = rf_model.score(X_test, y_test)
            precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], rf_predictions[:, i])
            f1score[i] = f1_score(y_test[:, i], rf_predictions[:, i])

            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], rf_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        for i in range(0, N):

            plt.figure(figsize=(8,8))
            plt.plot(recall[i], precision[i], lw=3, label='Class {}'.format(i))
            plt.ylabel('Recall', fontsize=20)
            plt.xlabel('Precision', fontsize=20)
            plt.title('Random Forest Classifier (Precision Recall Curve)', fontsize=20)
            plt.legend(fontsize=15)

            bytes_image1[i] = create_io_obj()

            plt.figure(figsize=(8,8))
            plt.plot(fpr[i], tpr[i], lw=3, label='AUC {0}; Class {1}'.format(roc_auc[i], i))
            plt.plot([0,1], [0, 1], 'k--')
            plt.ylabel('True Positive Rate (TPR)', fontsize=20)
            plt.xlabel('False Positive Rate (FPR)', fontsize=20)
            plt.title('Random Forest Classifier (ROC Curve)', fontsize=20)
            plt.legend(fontsize=15)

            bytes_image2[i] = create_io_obj()


        return bytes_image1,  bytes_image2

    elif cf_matrix == True:

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        # Grid Search
        
        param_grid  = {'n_estimators':[10, 20, 30], 'max_depth':[2, 3, 4, 5], 'min_samples_split':[5, 10, 15]}
        # param_grid  = {'n_estimators':[10], 'max_depth':[2], 'min_samples_split':[5]}

        rf_model = RandomForestClassifier(random_state=0)
        grid  =  GridSearchCV(rf_model, param_grid, cv=5, refit=True, verbose=2)
        grid.fit(X_train, y_train)

        rf_model = RandomForestClassifier(n_estimators=grid.best_params_['n_estimators'],\
                max_depth=grid.best_params_['max_depth'],\
                min_samples_split=grid.best_params_['min_samples_split'],\
                random_state=0).fit(X_train, y_train)

        # Classificaiton Report and Confusion Matrix 

        accuracy = rf_model.score(X_test, y_test)

        rf_predictions = rf_model.predict(X_test)

        
        # Plot Confusion Matrix
        ax = plot_confusion_matrix(y_test, rf_predictions)

        bytes_image = create_io_obj()
       
        return bytes_image 
