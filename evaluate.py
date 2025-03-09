import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn import svm,tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
import warnings

from collections import Counter

warnings.filterwarnings("ignore")

def supervised_model_training(x_train, y_train,
                              x_test, y_test, problem_type,
                              continuous_columns, model_name):
    '''
    Train and evaluates commonly used ML models

    Outputs:
    1) List of metrics contatining accuracy, auc, and f1-score of trained ML model
    '''

    if model_name == 'lr':
        model  = LogisticRegression(random_state=42,max_iter=500)
    elif model_name == 'dt':
        model  = tree.DecisionTreeClassifier(random_state=42)
    elif model_name == 'rf':
        model = RandomForestClassifier(random_state=42)
    elif model_name == "l_reg":
        model = LinearRegression()
    elif model_name == "lasso":
        model = Lasso(random_state = 42)
    elif model_name == "B_ridge":
        model = BayesianRidge()

    # In case of multi-class classification AUC and F1-SCORES are computed using weighted averages across all distinct labels
    if problem_type == "Classification":
        y_train = y_train.astype('str')
        y_test = y_test.astype('str')
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        if len(y_train.unique()) > 2:
            predict = model.predict_proba(x_test)
            acc = metrics.accuracy_score(y_test,pred)*100
            auc = metrics.roc_auc_score(y_test, predict,average="weighted",multi_class="ovr")
            f1_score = metrics.precision_recall_fscore_support(y_test, pred,average="weighted")[2]
            return [acc, auc, f1_score]

        else:
            predict = model.predict_proba(x_test)[:,1]
            acc = metrics.accuracy_score(y_test,pred)*100
            auc = metrics.roc_auc_score(y_test, predict)
            f1_score = metrics.precision_recall_fscore_support(y_test,pred)[2].mean()
            return [acc, auc, f1_score]

    else:
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        mape = metrics.mean_absolute_percentage_error(y_test,pred)
        mae = metrics.mean_absolute_error(y_test, pred)
        r2_score = metrics.r2_score(y_test,pred)
        return [mape, mae, r2_score]


def get_utility_metrics(real_train_path, real_test_path, fake_paths,
                        problem_type,
                        target_col, continuous_columns_wo_target, category_columns_wo_target,
                        scaler = 'MinMax', classifiers = ['lr', 'dt', 'rf']):
    '''
    Return ML utility metrics

    Outputs:
    1) results
    '''
    data_real_train = pd.read_csv(real_train_path)
    data_real_test = pd.read_csv(real_test_path)
    len_real_train = len(data_real_train)

    data_columns = list(data_real_train.columns)
    data_columns.remove(target_col)

    X_train_real = data_real_train[data_columns]
    y_train_real = data_real_train[target_col]

    X_test_real = data_real_test[data_columns]
    y_test_real = data_real_test[target_col]

    for col in category_columns_wo_target:
        col_train_categories = set(X_train_real[col].unique())
        col_test_categories = set(X_test_real[col].unique())

        test_unseen_categories = col_test_categories - col_train_categories

        if len(test_unseen_categories) != 0:
            col_train_count_items = Counter(X_train_real[col])
            train_most_freq_category = col_train_count_items.most_common(n = 1)[0][0]

            for unseen_category in test_unseen_categories:
                X_test_real[col].replace(unseen_category, train_most_freq_category, inplace = True)

    for col in category_columns_wo_target:
        le_real = LabelEncoder()
        le_real.fit(X_train_real[col])
        X_train_real[col] = le_real.transform(X_train_real[col])
        X_test_real[col] = le_real.transform(X_test_real[col])

    if scaler == 'MinMax':
        scaler_real = MinMaxScaler()
    else:
        scaler_real = StandardScaler()

    scaler_real.fit(X_train_real[continuous_columns_wo_target])
    X_train_real[continuous_columns_wo_target] = scaler_real.transform(X_train_real[continuous_columns_wo_target])
    X_test_real[continuous_columns_wo_target] = scaler_real.transform(X_test_real[continuous_columns_wo_target])

    all_real_results = []
    for classifier in classifiers:
        real_results = supervised_model_training(X_train_real, y_train_real, X_test_real, y_test_real, problem_type, continuous_columns_wo_target, classifier)
        all_real_results.append(real_results)

    all_fake_results = []

    for fake_path in fake_paths:
        data_fake_train = pd.read_csv(fake_path)
        data_real_test = pd.read_csv(real_test_path)
        data_fake_train = data_fake_train.iloc[:len_real_train]
        data_real_test = data_real_test.iloc[:len_real_train]

        data_columns = list(data_fake_train.columns)
        data_columns.remove(target_col)

        X_train_fake = data_fake_train[data_columns]
        y_train_fake = data_fake_train[target_col]

        X_test_real = data_real_test[data_columns]
        y_test_real = data_real_test[target_col]

        for col in category_columns_wo_target:
            col_train_categories = set(X_train_fake[col].unique())
            col_test_categories = set(X_test_real[col].unique())

            test_unseen_categories = col_test_categories - col_train_categories

            if len(test_unseen_categories) != 0:
                col_train_count_items = Counter(X_train_fake[col])
                train_most_freq_category = col_train_count_items.most_common(n = 1)[0][0]

                for unseen_category in test_unseen_categories:
                    X_test_real[col].replace(unseen_category, train_most_freq_category, inplace = True)

        for col in category_columns_wo_target:
            le_fake = LabelEncoder()
            le_fake.fit(X_train_fake[col])
            X_train_fake[col] = le_fake.transform(X_train_fake[col])
            X_test_real[col] = le_fake.transform(X_test_real[col])

        if scaler == 'MinMax':
            scaler_fake = MinMaxScaler()
        else:
            scaler_fake = StandardScaler()

        scaler_fake.fit(X_train_fake[continuous_columns_wo_target])
        X_train_fake[continuous_columns_wo_target] = scaler_fake.transform(X_train_fake[continuous_columns_wo_target])
        X_test_real[continuous_columns_wo_target] = scaler_fake.transform(X_test_real[continuous_columns_wo_target])

        all_classifier_fake_results = []
        for classifier in classifiers:
            fake_results = supervised_model_training(X_train_fake, y_train_fake, X_test_real, y_test_real, problem_type, continuous_columns_wo_target, classifier)

            all_classifier_fake_results.append(fake_results)

        all_fake_results.append(all_classifier_fake_results)

    return all_real_results, all_fake_results


def get_utility_metrics_training(data_real_train, data_real_test, data_fake_train,
                                problem_type,
                                target_col, continuous_columns_wo_target, category_columns_wo_target,
                                scaler = 'MinMax', classifiers = ['lr', 'dt', 'rf']):
    '''
    Return ML utility metrics

    Outputs:
    1) results
    '''
    len_real_train = len(data_real_train)

    data_columns = list(data_real_train.columns)
    data_columns.remove(target_col)

    X_train_real = data_real_train[data_columns]
    y_train_real = data_real_train[target_col]

    X_test_real = data_real_test[data_columns]
    y_test_real = data_real_test[target_col]

    for col in category_columns_wo_target:
        col_train_categories = set(X_train_real[col].unique())
        col_test_categories = set(X_test_real[col].unique())

        test_unseen_categories = col_test_categories - col_train_categories

        if len(test_unseen_categories) != 0:
            col_train_count_items = Counter(X_train_real[col])
            train_most_freq_category = col_train_count_items.most_common(n = 1)[0][0]

            for unseen_category in test_unseen_categories:
                X_test_real[col].replace(unseen_category, train_most_freq_category, inplace = True)

    for col in category_columns_wo_target:
        le_real = LabelEncoder()
        le_real.fit(X_train_real[col])
        X_train_real[col] = le_real.transform(X_train_real[col])
        X_test_real[col] = le_real.transform(X_test_real[col])

    if scaler == 'MinMax':
        scaler_real = MinMaxScaler()
    else:
        scaler_real = StandardScaler()

    scaler_real.fit(X_train_real[continuous_columns_wo_target])
    X_train_real[continuous_columns_wo_target] = scaler_real.transform(X_train_real[continuous_columns_wo_target])
    X_test_real[continuous_columns_wo_target] = scaler_real.transform(X_test_real[continuous_columns_wo_target])

    all_real_results = []
    for classifier in classifiers:
        real_results = supervised_model_training(X_train_real, y_train_real, X_test_real, y_test_real, problem_type, continuous_columns_wo_target, classifier)
        all_real_results.append(real_results)

    data_fake_train = data_fake_train.iloc[:len_real_train]
    data_real_test = data_real_test.iloc[:len_real_train]

    data_columns = list(data_fake_train.columns)
    data_columns.remove(target_col)

    X_train_fake = data_fake_train[data_columns]
    y_train_fake = data_fake_train[target_col]

    X_test_real = data_real_test[data_columns]
    y_test_real = data_real_test[target_col]

    for col in category_columns_wo_target:
        col_train_categories = set(X_train_fake[col].unique())
        col_test_categories = set(X_test_real[col].unique())

        test_unseen_categories = col_test_categories - col_train_categories

        if len(test_unseen_categories) != 0:
            col_train_count_items = Counter(X_train_fake[col])
            train_most_freq_category = col_train_count_items.most_common(n = 1)[0][0]

            for unseen_category in test_unseen_categories:
                X_test_real[col].replace(unseen_category, train_most_freq_category, inplace = True)

    for col in category_columns_wo_target:
        le_fake = LabelEncoder()
        le_fake.fit(X_train_fake[col])
        X_train_fake[col] = le_fake.transform(X_train_fake[col])
        X_test_real[col] = le_fake.transform(X_test_real[col])

    if scaler == 'MinMax':
        scaler_fake = MinMaxScaler()
    else:
        scaler_fake = StandardScaler()

    scaler_fake.fit(X_train_fake[continuous_columns_wo_target])
    X_train_fake[continuous_columns_wo_target] = scaler_fake.transform(X_train_fake[continuous_columns_wo_target])
    X_test_real[continuous_columns_wo_target] = scaler_fake.transform(X_test_real[continuous_columns_wo_target])

    all_classifier_fake_results = []

    for classifier in classifiers:
        fake_results = supervised_model_training(X_train_fake, y_train_fake, X_test_real, y_test_real, problem_type, continuous_columns_wo_target, classifier)

        all_classifier_fake_results.append(fake_results)
    
    result = np.array(all_real_results) - np.array(all_classifier_fake_results)
    print("result:{}".format(result))
    
    if problem_type == 'Classification':
        result_summary = np.mean(result[:, 0])
    else:
        result_summary = np.mean(result[:, 0]) * -1

    return result_summary

def privacy_metrics(real_path,fake_path,category_columns_wo_target,data_percent=15):
    """
    Returns privacy metrics

    Inputs:
    1) real_path -> path to real data
    2) fake_path -> path to corresponding synthetic data
    3) data_percent -> percentage of data to be sampled from real and synthetic datasets for computing privacy metrics

    Outputs:
    1) List containing the 5th percentile distance to closest record (DCR) between real and synthetic as well as within real and synthetic datasets
    along with 5th percentile of nearest neighbour distance ratio (NNDR) between real and synthetic as well as within real and synthetic datasets

    """

    # Loading real and synthetic datasets and removing duplicates if any
    real = pd.read_csv(real_path).drop_duplicates(keep=False)
    fake = pd.read_csv(fake_path).drop_duplicates(keep=False)

    for col in category_columns_wo_target:
        le_real = LabelEncoder()
        le_real.fit(real[col])
        real[col] = le_real.transform(real[col])
        fake[col] = le_real.transform(fake[col])

    # Sampling smaller sets of real and synthetic data to reduce the time complexity of the evaluation
    real_sampled = real.sample(n=int(len(real)*(.01*data_percent)), random_state=42).to_numpy()
    fake_sampled = fake.sample(n=int(len(fake)*(.01*data_percent)), random_state=42).to_numpy()

    # Scaling real and synthetic data samples
    scalerR = StandardScaler()
    scalerR.fit(real_sampled)
    scalerF = StandardScaler()
    scalerF.fit(fake_sampled)
    df_real_scaled = scalerR.transform(real_sampled)
    df_fake_scaled = scalerF.transform(fake_sampled)

    # Computing pair-wise distances between real and synthetic
    dist_rf = metrics.pairwise_distances(df_real_scaled, Y=df_fake_scaled, metric='minkowski', n_jobs=-1)
    # Computing pair-wise distances within real
    dist_rr = metrics.pairwise_distances(df_real_scaled, Y=None, metric='minkowski', n_jobs=-1)
    # Computing pair-wise distances within synthetic
    dist_ff = metrics.pairwise_distances(df_fake_scaled, Y=None, metric='minkowski', n_jobs=-1)

    # Removes distances of data points to themselves to avoid 0s within real and synthetic
    rd_dist_rr = dist_rr[~np.eye(dist_rr.shape[0],dtype=bool)].reshape(dist_rr.shape[0],-1)
    rd_dist_ff = dist_ff[~np.eye(dist_ff.shape[0],dtype=bool)].reshape(dist_ff.shape[0],-1)

    # Computing first and second smallest nearest neighbour distances between real and synthetic
    smallest_two_indexes_rf = [dist_rf[i].argsort()[:2] for i in range(len(dist_rf))]
    smallest_two_rf = [dist_rf[i][smallest_two_indexes_rf[i]] for i in range(len(dist_rf))]
    # Computing first and second smallest nearest neighbour distances within real
    smallest_two_indexes_rr = [rd_dist_rr[i].argsort()[:2] for i in range(len(rd_dist_rr))]
    smallest_two_rr = [rd_dist_rr[i][smallest_two_indexes_rr[i]] for i in range(len(rd_dist_rr))]
    # Computing first and second smallest nearest neighbour distances within synthetic
    smallest_two_indexes_ff = [rd_dist_ff[i].argsort()[:2] for i in range(len(rd_dist_ff))]
    smallest_two_ff = [rd_dist_ff[i][smallest_two_indexes_ff[i]] for i in range(len(rd_dist_ff))]


    # Computing 5th percentiles for DCR and NNDR between and within real and synthetic datasets
    min_dist_rf = np.array([i[0] for i in smallest_two_rf])
    fifth_perc_rf = np.percentile(min_dist_rf,5)
    min_dist_rr = np.array([i[0] for i in smallest_two_rr])
    fifth_perc_rr = np.percentile(min_dist_rr,5)
    min_dist_ff = np.array([i[0] for i in smallest_two_ff])
    fifth_perc_ff = np.percentile(min_dist_ff,5)
    nn_ratio_rf = np.array([i[0]/i[1] for i in smallest_two_rf])
    nn_fifth_perc_rf = np.percentile(nn_ratio_rf,5)
    nn_ratio_rr = np.array([i[0]/i[1] for i in smallest_two_rr])
    nn_fifth_perc_rr = np.percentile(nn_ratio_rr,5)
    nn_ratio_ff = np.array([i[0]/i[1] for i in smallest_two_ff])
    nn_fifth_perc_ff = np.percentile(nn_ratio_ff,5)

    return np.array([fifth_perc_rf,fifth_perc_rr,fifth_perc_ff,nn_fifth_perc_rf,nn_fifth_perc_rr,nn_fifth_perc_ff]).reshape(1,6), min_dist_rf