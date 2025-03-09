import os
import random

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import LabelEncoder

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class Preprocessor():
    def __init__(self, data, continuous_columns:list, mixed:dict, categorical_columns:list,
                 skew_columns:list, integer_columns:list, num_mode_list:list):
        '''
            data(pd.DataFrame): Data have to transform
            continuous_columns(list): continuous feature column names
            mixed(dict):
                key: mixed feature column names
                item: mixed feature categorical items
            categorical_columns(list): categorical feature column names
            skew_columns(list): skewed feature column names
            integer_columns(list): continuous_columns which all elements are integer
        '''
        self.data = data

        self.continuous_columns = continuous_columns
        self.categorical_columns = categorical_columns
        self.skew_columns = skew_columns
        self.integer_columns = integer_columns
        self.mixed = mixed

        self.continuous_transformer_list = []
        for num_mode in num_mode_list:
            self.continuous_transformer_list.append(MixedEncoder(num_mode))

        self.categorical_transformer = LabelEncoder()

        self.num_mode_list = num_mode_list
        self.lower_bounds = {}

        self.metadata = None

        # dealing with skewed features by applying log transformation
        if len(self.skew_columns) != 0:
            for skew_col in self.skew_columns:
                # Value added to apply to non-positive numeric values
                eps = 1

                lower = np.min(self.data[skew_col].values)
                self.lower_bounds[skew_col] = lower

                if lower > 0:
                    self.data[skew_col] = self.data[skew_col].apply(lambda x:np.log(x))
                elif lower == 0:
                    self.data[skew_col] = self.data[skew_col].apply(lambda x:np.log(x+eps))
                else: # negative
                    self.data[skew_col] = self.data[skew_col].apply(lambda x:np.log(x-lower+eps))

    def inverse_prep(self, data, eps=1):
        # inverse skew columns
        for skew_col in self.skew_columns:
            lower_bound = self.lower_bounds[skew_col]

            if lower_bound > 0:
                data[skew_col].apply(lambda x:np.exp(x))
            elif lower_bound == 0:
                data[skew_col].apply(lambda x:np.ceil(np.exp(x)-eps))
            else:
                data[skew_col].apply(lambda x:np.exp(x)-eps+lower_bound)

        # inverse integer columns
        for int_col in self.integer_columns:
            data[int_col] = (np.round(data[int_col].values))
            data[int_col] = data[int_col].astype(int)

        return data

    def fit_transform(self, data):
        '''
        Transform human-readable data into meaningful features

            args:
                data(pd.DataFrame): Data to transform

            returns:
                pd.DataFrame: Model Features
        '''
        num_cols = data.shape[1]

        transformed_data = {}
        details = []
        start_embed_dim = 0
        total_embed_dim = 0
        data_columns = data.columns
        start_embed_dim_dict = {}

        cnt = 0
        for i in data.columns:
            if i in self.continuous_columns:
                # preprocessing for continuous columns
                column_data = data[i].values.reshape([-1, 1])

                if i in self.mixed.keys():
                    features, prob_argmax, probs, means, stds, model, output_info = self.continuous_transformer_list[cnt].fit_transform(column_data, self.mixed[i])
                    details.append({
                        "type":"mixed",
                        "col_name":i,
                        "means": means,
                        "stds": stds,
                        "n": features.shape[1] + prob_argmax.shape[1],
                        "model": model,
                        "num_categories": len(self.mixed[i]),
                        "categories": self.mixed[i],
                        "output_info": output_info,
                        "start_embed_dim": start_embed_dim
                    })
                    start_embed_dim = start_embed_dim + (features.shape[1] + prob_argmax.shape[1])

                else:
                    features, prob_argmax, probs, means, stds, model, output_info = self.continuous_transformer_list[cnt].fit_transform(column_data, [])
                    details.append({
                        "type": "continuous",
                        "col_name": i,
                        "means": means,
                        "stds": stds,
                        "n": 1 + prob_argmax.shape[1],
                        "model": model,
                        "output_info": output_info,
                        "start_embed_dim": start_embed_dim
                    })
                    start_embed_dim = start_embed_dim + (1 + prob_argmax.shape[1])

                transformed_data['f%02d' % i] = np.concatenate((features, prob_argmax), axis=1)

                cnt+=1

            else:
                # preprocessing for categorical columns
                column_data = data[i].astype(str).values

                features = self.categorical_transformer.fit_transform(column_data)
                transformed_data['f%02d' % i] = features.reshape([-1, 1])

                mapping = self.categorical_transformer.classes_

                output_info = [(len(data[i].unique()), 'softmax')]

                details.append({
                    "type": "category",
                    "col_name": i,
                    "mapping": mapping,
                    "n": mapping.shape[0],
                    "output_info": output_info,
                    "start_embed_dim": start_embed_dim
                })
                start_embed_dim = start_embed_dim + (len(data[i].unique()))

        total_embed_dim = start_embed_dim

        for detail in details:
            col_name = detail["col_name"]
            start_embed_dim = detail["start_embed_dim"]

            start_embed_dim_dict[col_name] = start_embed_dim

        metadata = {
            "columns": data_columns,
            "num_features": num_cols,
            "details": details,
            "start_embed_dim": start_embed_dim_dict,
            "total_embed_dim": total_embed_dim
        }
        self.metadata = metadata

        return transformed_data

    def transform(self, data):
        '''
        Transform the given dataframe without generating new metadata

            args:
                data(pd.DataFrame): Data to fit the object
        '''
        transformed_data = {}

        cnt = 0
        for i in data.columns:
            if i in self.continuous_columns:
                # preprocessing for continuous columns
                column_data = data[i].values.reshape([-1, 1])

                if i in self.mixed.keys():
                    features, prob_argmax = self.continuous_transformer_list[cnt].fit_transform(column_data, self.mixed[i])

                else:
                    features, prob_argmax = self.continuous_transformer_list[cnt].fit_transform(column_data, [])

                transformed_data['f%02d' % i] = np.concatenate((features, prob_argmax), axis=1)

                cnt+=1
            else:
                # preprocessing for categorical columns
                column_data = data[i].astype(str).values

                features = self.categorical_transformer.fit_transform(column_data)
                transformed_data['f%02d' % i] = features.reshape([-1, 1])

        return transformed_data

    def fit(self, data):
        '''
        Initialize the internal state of the object using data

        args:
            data(pd.DataFrame): Data to fit the object

        No return
        '''
        self.fit_transform(data)

    def reverse_transform(self, data):
        '''
        Transform numerical(meaningful) features back into human-readable data

        args:
            data(pd.DataFrame): Data to transform

        returns:
            pd.DataFrame: human-readable data
        '''
        table = []
        list_columns = []

        cnt = 0
        for column_metadata in self.metadata['details']:
            column_name = column_metadata['col_name']
            column_data = data['f%02d' % column_name]

            if column_metadata['type'] == 'mixed':
                column = self.continuous_transformer_list[cnt].inverse_transform(column_data, column_metadata)
                cnt += 1

            elif column_metadata['type'] == 'continuous':
                column = self.continuous_transformer_list[cnt].inverse_transform(column_data, column_metadata)
                cnt += 1

            elif column_metadata['type'] == 'category':
                self.categorical_transformer.classes_ = column_metadata['mapping']
                column = self.categorical_transformer.inverse_transform(
                    column_data.ravel().astype(np.int32)
                )

            list_columns.append(column_name)
            table.append(column)

        result = pd.DataFrame(dict(enumerate(table)))
        result.columns = list_columns
        return result

def continuous_embed_to_numeric(in_tensor, target_idx, metadata, device, batch_size=256):
    col_detail = metadata['details'][target_idx]
    means = col_detail['means']
    stds = col_detail['stds']

    means_tensor_mold = torch.tensor(means, dtype=torch.float32).repeat(batch_size).to(device)
    stds_tensor_mold = torch.tensor(stds, dtype=torch.float32).repeat(batch_size).to(device)

    col_alpha = in_tensor[:, 0]
    col_beta = in_tensor[:, 1:]
    col_beta_argmax = torch.argmax(col_beta, dim=1, keepdim=True)

    means_tensor = means_tensor_mold[col_beta_argmax]
    stds_tensor = stds_tensor_mold[col_beta_argmax]
    col_alpha_ = col_alpha.unsqueeze(1)

    numeric_col = means_tensor + stds_tensor * col_alpha_ * 4

    return numeric_col


class MixedEncoder():
    '''
    Reversible transformation for Mixed type data and Continuous data

    How?
        Mixed type variable
            scalar(input) ->    case 1. Categorical
                                        [0] + [0, ... 0, 1, 0, ..., 0]
                                case 2. Continuous
                                        [Normalized Value] + [0, ... 0, 1, 0, ..., 0]
        Continuous type variable
            scalar(input) ->    [Normalized Value] + [0, ... 0, 1, 0, ..., 0]

    args:
        num_modes, eps
    '''
    def __init__(self, num_modes = 5, eps=0.005):
        self.num_modes = num_modes
        self.eps = eps

    def fit_transform(self, col_data, categories):
        '''
        args:
            data(numpy.ndarray): Values to cluster in array of shape (n, 1)
            categories(list): if the feature column is mixed variable, it contains multiple categories

        returns:
            tuple[numpy.ndarray, numpy.ndarray, list, list]:
                Tuple containing the features, probabilities, averages and stds of given data
        '''
        if len(categories) != 0: # mixed type
            model = GaussianMixture(self.num_modes)

            # gmm should trained with data which is not contain categories
            filter_arr = []
            for element in col_data:
                if element not in categories:
                    filter_arr.append(True)
                else:
                    filter_arr.append(False)

            model.fit(col_data[filter_arr])

            means = model.means_.reshape((1, self.num_modes))
            stds = np.sqrt(model.covariances_).reshape((1, self.num_modes))

            features = (col_data - means) / (4 * stds)
            prob_argmax = np.zeros((col_data.shape[0], self.num_modes+len(categories)))

            probs = model.predict_proba(col_data)
            argmax = np.argmax(probs, axis = 1)

            idx = np.arange(len(features))
            features = features[idx, argmax].reshape([-1, 1])
            features = np.clip(features, -0.99, 0.99)

            for i, filter_ in enumerate(filter_arr):
                if filter_:
                    # not in categories
                    prob_argmax[i, len(categories) + argmax[i]] = 1
                else:
                    # in categories
                    prob_argmax[i, categories.index(col_data[i])] = 1
                    features[i]=0

            output_info = [(1, 'tanh'), (self.num_modes + len(categories), 'softmax')]

        else: # normal continuous type
            # model = GaussianMixture(self.num_modes)
            model = BayesianGaussianMixture(n_components = self.num_modes)

            model.fit(col_data)

            means = model.means_.reshape((1, self.num_modes))
            stds = np.sqrt(model.covariances_).reshape((1, self.num_modes))

            features = (col_data - means) / (4 * stds)
            prob_argmax = np.zeros_like(features)

            probs = model.predict_proba(col_data)
            argmax = np.argmax(probs, axis = 1)

            idx = np.arange(len(features))
            features = features[idx, argmax].reshape([-1, 1])
            features = np.clip(features, -0.99, 0.99)

            prob_argmax[idx, argmax] = 1

            output_info = [(1, 'tanh'), (self.num_modes, 'softmax')]

        return features, prob_argmax, probs, list(means.flat), list(stds.flat), model, output_info

    def transform(self, col_data, info):
        '''
        Transform

        args:
            col_data(numpy.ndarray): Values to cluster in array of shape (n, 1)
            info(dict): column's metadat
        '''
        if info["type"] == "mixed":
            # mixed type
            model = info["model"]
            means = info["means"]
            stds = info["stds"]
            categories = info["categories"]

            probs = model.predict_proba(col_data)
            prob_argmax = np.zeros((col_data.shape[0], self.num_modes+len(categories)))

            probs = model.predict_proba(col_data)
            argmax = np.argmax(probs, axis = 1)

            idx = np.arange(len(features))
            features = features[idx, argmax].reshape([-1, 1])
            features = np.clip(features, -0.99, 0.99)

            filter_arr = []
            for element in col_data:
                if element not in categories:
                    filter_arr.append(True)
                else:
                    filter_arr.append(False)

            for i, filter_ in enumerate(filter_arr):
                if filter_:
                    # not in categories
                    prob_argmax[i, len(categories) + argmax[i]] = 1
                else:
                    # in categories
                    prob_argmax[i, categories.index(col_data[i])] = 1
                    features[i]=0

        else:
            # normal continuous
            model = info["model"]
            means = info["means"]
            stds = info["stds"]

            features = (col_data - means) / (4 * stds)
            prob_argmax = np.zeros_like(features)

            probs = model.predict_proba(col_data)
            argmax = np.argmax(probs, axis = 1)

            idx = np.arange(len(features))
            features = features[idx, argmax].reshape([-1, 1])
            features = np.clip(features, -0.99, 0.99)

            prob_argmax[idx, argmax] = 1

        return features, prob_argmax


    def inverse_transform(self, col_data, info):
        '''
        Reverse the clustering of values

        args:
            col_data(numpy.ndarray): transformed col_data to restore.
            info(dict): metadata

        returns:
            numpy.ndarray: values in the original space
        '''
        if info["type"] == "mixed":
            num_categories = info["num_categories"]
            categories = info["categories"]

            features = col_data[:, 0]
            prob_argmax = col_data[:, 1:]

            p_argmax = np.argmax(prob_argmax, axis = 1)

            mean = np.asarray(info['means'])
            std = np.asarray(info['stds'])

            filter_arr = p_argmax < num_categories
            # if filter_arr element is True -> categories

            inverse_features = np.zeros_like(features)

            for idx, filter_ in enumerate(filter_arr):
                if filter_:
                    # treat for categories
                    inverse_features[idx] = categories[p_argmax[idx]]

                else:
                    # treat for continuous cases
                    inverse_features[idx] = features[idx] * 4 * std[p_argmax[idx] - num_categories] \
                      + mean[p_argmax[idx] - num_categories]

            return inverse_features

        else:
            features = col_data[:, 0]
            prob_argmax = col_data[:, 1:]

            p_argmax = np.argmax(prob_argmax, axis = 1)

            mean = np.asarray(info['means'])
            std = np.asarray(info['stds'])

            select_mean = mean[p_argmax]
            select_std = std[p_argmax]

            return features * 4 * select_std + select_mean

class CategoricalEmbedding:
    '''
        Reversible embedding for preprocessed data
    '''
    def __init__(self, metadata):
        '''
        args:
            metadata:
        '''
        self.metadata = metadata

    def embed(self, preprocessed_data, add_noise = False):
        '''
        categorical data -> one hot encoding

        args:
        preprocessed_data(pd.DataFrame): Preprocessed dataframe
            *   Continuous features
            *   Categorical features: label of categories; scalar
        add_noise(bool)
            *   True    : add noise to one hot encoding of categorical featueres
            *   False   : not add noise to one hot encoding

        returns:
            embedded_data(dict)
            output_info
        '''
        embedded_data = {}
        categori_embed_inform = []

        for i in range(self.metadata['num_features']):
            for detail in self.metadata['details']:
                if detail['col_name'] == i:
                    column_detail = detail
                    break

            column_type = column_detail['type']
            column_data = preprocessed_data['f%02d' % i]

            if column_type == 'category':
                # one hot encoding
                one_hot_column_data = np.eye(len(np.unique(column_data)))[column_data]
                one_hot_column_data = one_hot_column_data.reshape(-1, len(np.unique(column_data)))

                if add_noise:
                    # add noise
                    noise = np.random.uniform(0, 1 / (len(np.unique(column_data)*50)), size= (len(column_data), len(np.unique(column_data))))
                    one_hot_column_data = one_hot_column_data + noise

                # normalize
                sum_column = np.sum(one_hot_column_data, axis = 1)
                sum_column = np.reshape(sum_column, (len(sum_column), 1))
                norm_column_data = one_hot_column_data / sum_column

                embedded_data['f%02d' % i] = norm_column_data

                # save categori_embed_inform
                embed_inform = {'col_name':i,
                                'embed_dim':norm_column_data.shape[1]}
                categori_embed_inform.append(embed_inform)

            else:
                embedded_data['f%02d' % i] = column_data

        self.metadata['categori_embed_inform'] = categori_embed_inform

        return embedded_data

def before_categorical_embedding(embeded_data, metadata):
    '''
    find original form of embeded_data before categorical embedding

    args:
        embeded_data (dict)
        - key: f{feature_name}, item: np.array
        metadata
        - details

    returns:
        bef_embed_data (dict)
        - key: f{feature_name}, item: np.array
    '''
    bef_embed_data = {}

    for detail in metadata['details']:
        feature_name = detail['col_name']
        feature_type = detail['type']
        feature_data = embeded_data['f%02d' % feature_name]

        if feature_type == 'category':
            # find maximum idx
            bef_feature_data = np.argmax(feature_data, axis = 1)

            bef_embed_data['f%02d' % feature_name] = bef_feature_data
        else:
            bef_embed_data['f%02d' % feature_name] = feature_data

    return bef_embed_data

class MergeDataset:
    def __init__(self, dataset, metadata):
        self.dataset = dataset
        self.metadata = metadata

    def merge(self):
        arrays = []

        for i in range(self.metadata['num_features']):
            arrays.append(self.dataset['f%02d' % i])

        merge_array = np.concatenate(arrays, axis = 1)

        return merge_array

class TabularDataset(Dataset):
    '''
    Declare custom dataset module for tabular dataset handling
    '''
    def __init__(self, data_tensor):
        self.x_data = data_tensor
        self.y_data = torch.ones(data_tensor.shape[0])

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]

        return x, y

def unpack_tensor(input_tensor, metadata):
    '''
    Unpack input tensor to each feature arrays

    args:
        input_tensor(torch.tensor)
        - shape: (num_samples, total_feature_dims)
        metadata
        - details

    returns:
        dictionary - key: f{col_name}, item: np.array
    '''
    input_tensor_cpu = input_tensor.to('cpu')

    unpack_results = {}

    col_idx = 0

    for detail in metadata['details']:
        col_name = detail['col_name']
        col_type = detail['type']

        if col_type == 'value':
            dim = detail['n'] + 1
        else:
            dim = detail['n']

        unpack_results['f%02d' % col_name] = np.array(input_tensor_cpu[:, col_idx:col_idx + dim].detach())
        col_idx = col_idx + dim

    return unpack_results

class DataSampler():
    '''
    DataSampler samples the conditional vector and corresponding data.
    '''

    def __init__(self, data:pd.DataFrame, metadata:dict):
        '''
            input:
                data(pd.DataFrame)
                metadata(metadata)
        '''
        # gather categorical column info
        self.data = data
        self.metadata = metadata

        # step1. organize categorical column info
        '''
            self.cate_prob_info
                key: categorical column name
                item: categorical column pmf list
            self.cate_item_info
                key: categorical column name
                item: categorical column item list
        '''
        self.cate_prob_info = {}
        self.cate_prob_info_valid = {}
        self.cate_item_info = {}
        self.n_categorical_columns = 0
        self.cate_colname_list = []
        self.cate_column_item = {}

        largest_embed_dim = 0

        for detail in metadata['details']:
            type = detail['type']
            col_name = detail['col_name']

            if type == 'category':
                # calculate pmf
                col_prob = self.data[col_name].value_counts()/len(self.data)
                log_col_prob = np.log(col_prob) / np.sum(np.log(col_prob))

                # save categorical column's items
                self.cate_column_item[col_name] = self.data[col_name].value_counts().index

                # store info
                self.cate_prob_info[col_name] = log_col_prob
                self.cate_prob_info_valid[col_name] = col_prob

                self.cate_item_info[col_name] = self.data[col_name].value_counts().keys()

                self.cate_colname_list.append(col_name)

                self.n_categorical_columns = self.n_categorical_columns + 1

        # step2. organize data to parsing more quickly in sample_condvec_data method
        # cat_col_item[col_name][item_name] : list of data corresponding data index
        self.cate_col_item_index = {}

        for cat_col in self.cate_colname_list:
            self.cate_col_item_index[cat_col] = []

            cat_col_item_idx = {}

            for col_item in self.cate_column_item[cat_col]:
                correct_idx = data[data[cat_col] == col_item].index
                cat_col_item_idx[col_item] = correct_idx

            self.cate_col_item_index[cat_col] = cat_col_item_idx

    def sample_condvec_data_train(self, cond_dim, batch, target_col):
        # target_tensor = torch.zeros(size=(batch, len(self.cate_col_item_index[target_col])))

        cond_tensor = torch.tensor(np.zeros(shape=(batch, cond_dim)), dtype=torch.float32)
        cond_random_sample = np.zeros(shape=(batch))

        # step 1. choose random categorical column uniformly
        random_chosen_columns = np.random.randint(low=0, high=self.n_categorical_columns,
                                                  size=(batch, 1))
        r = np.expand_dims(np.random.rand(batch), axis=1)

        # step 2. chooose random category in the chosen categorical column by using pmf
        for i in range(batch):
            rand_colname = self.cate_colname_list[random_chosen_columns[i][0]]
            # rand_col_prob = self.cate_prob_info[rand_colname]
            rand_col_prob = self.cate_prob_info_valid[rand_colname]

            # choose random category
            rand_cat = (rand_col_prob.cumsum() > r[i][0]).argmax()
            start_embed_dim = self.metadata['start_embed_dim'][rand_colname]
            rand_cat_embed_dim_idx = start_embed_dim + rand_cat

            # modify cond_tensor
            cond_tensor[i, rand_cat_embed_dim_idx] = 1

            # step 3. sample data
            sample_idx = random.choice(self.cate_col_item_index[rand_colname][self.cate_column_item[rand_colname][rand_cat]])

            cond_random_sample[i] = sample_idx

            # target_tensor[i][self.data[target_col][sample_idx]] = 1

        random_chosen_columns = torch.tensor(random_chosen_columns, dtype = torch.int32)

        return cond_tensor, random_chosen_columns, cond_random_sample

    def sample_condvec_data_generate(self, cond_dim, batch, target_col):
        # target_tensor = torch.zeros(size=(batch, len(self.cate_col_item_index[target_col])))

        cond_tensor = torch.tensor(np.zeros(shape=(batch, cond_dim)), dtype=torch.float32)
        cond_random_sample = np.zeros(shape=(batch))

        # step 1. choose random categorical column uniformly
        random_chosen_columns = np.random.randint(low=0, high=self.n_categorical_columns,
                                                  size=(batch, 1))
        r = np.expand_dims(np.random.rand(batch), axis=1)

        # step 2. chooose random category in the chosen categorical column by using pmf
        for i in range(batch):
            rand_colname = self.cate_colname_list[random_chosen_columns[i][0]]
            # rand_col_prob = self.cate_prob_info[rand_colname]
            rand_col_prob = self.cate_prob_info_valid[rand_colname]

            # choose random category
            rand_cat = (rand_col_prob.cumsum() > r[i][0]).argmax()
            start_embed_dim = self.metadata['start_embed_dim'][rand_colname]
            rand_cat_embed_dim_idx = start_embed_dim + rand_cat

            # modify cond_tensor
            cond_tensor[i, rand_cat_embed_dim_idx] = 1

            # step 3. sample data
            sample_idx = random.choice(self.cate_col_item_index[rand_colname][self.cate_column_item[rand_colname][rand_cat]])

            cond_random_sample[i] = sample_idx

            # target_tensor[i][self.data[target_col][sample_idx]] = 1

        random_chosen_columns = torch.tensor(random_chosen_columns, dtype = torch.int32)

        return cond_tensor, random_chosen_columns, cond_random_sample