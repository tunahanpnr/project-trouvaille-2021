from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def try_models(X, y):

    # prepare models
    models_ = []
    models_.append(('LR', LogisticRegression(max_iter=1000)))
    models_.append(('LDA', LinearDiscriminantAnalysis()))
    models_.append(('KNN', KNeighborsClassifier()))
    models_.append(('CART', DecisionTreeClassifier()))
    models_.append(('NB', GaussianNB()))
    models_.append(('SVM', SVC()))
    # evaluate each model in turn
    results = []
    names = []
    scoring = 'accuracy'
    for name, model in models_:
        kfold = model_selection.KFold(n_splits=10, shuffle=True)  # değişiklik dene
        cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f                  (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)