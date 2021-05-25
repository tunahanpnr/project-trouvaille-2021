import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def run_exps(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
    # prepare models
    dfs = []
    models_ = []
    models_.append(('LR', LogisticRegression(max_iter=1000)))
    models_.append(('RF', RandomForestClassifier()))
    models_.append(('LDA', LinearDiscriminantAnalysis()))
    models_.append(('KNN', KNeighborsClassifier()))
    models_.append(('CART', DecisionTreeClassifier()))
    # models_.append(('NB', GaussianNB()))
    models_.append(('SVM', SVC()))
    # evaluate each model in turn
    results = []
    names = []
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    target_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

    for name, model in models_:
        kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
        cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
        clf = model.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(name)
        print(classification_report(y_test, y_pred, target_names=target_names))

        results.append(cv_results)
        names.append(name)

        this_df = pd.DataFrame(cv_results)
        this_df['model'] = name
        dfs.append(this_df)

    final = pd.concat(dfs, ignore_index=True)

    # ------------------------------------------------------------------

    bootstraps = []
    for model in list(set(final.model.values)):
        model_df = final.loc[final.model == model]
        bootstrap = model_df.sample(n=30, replace=True)
        bootstraps.append(bootstrap)

    bootstrap_df = pd.concat(bootstraps, ignore_index=True)
    results_long = pd.melt(bootstrap_df, id_vars=['model'], var_name='metrics', value_name='values')
    time_metrics = ['fit_time', 'score_time']  # fit time metrics
    # PERFORMANCE METRICS
    results_long_nofit = results_long.loc[~results_long['metrics'].isin(time_metrics)]  # get df without fit data
    results_long_nofit = results_long_nofit.sort_values(by='values')
    # TIME METRICS
    results_long_fit = results_long.loc[results_long['metrics'].isin(time_metrics)]  # df with fit data
    results_long_fit = results_long_fit.sort_values(by='values')

    plt.figure(figsize=(20, 12))
    sns.set(font_scale=2.5)
    sns.boxplot(x="model", y="values", hue="metrics", data=results_long_nofit, palette="Set3")
    plt.legend()
    plt.title('Comparison of Model by Classification Metric')
    plt.savefig('./benchmark_models_performance.png', dpi=300)

    plt.figure(figsize=(20, 12))
    sns.set(font_scale=2.5)
    g = sns.boxplot(x="model", y="values", hue="metrics", data=results_long_fit, palette="Set3")
    plt.legend()
    plt.title('Comparison of Model by Fit and Score Time')
    plt.savefig('./benchmark_models_time.png', dpi=300)

    metrics = list(set(results_long_nofit.metrics.values))
    bootstrap_df.groupby(['model'])[metrics].agg([np.std, np.mean])

    time_metrics = list(set(results_long_fit.metrics.values))
    print(bootstrap_df.groupby(['model'])[time_metrics].agg([np.std, np.mean]))

    return final


