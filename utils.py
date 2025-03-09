import os
from tqdm import tqdm

import copy
import pandas as pd
import numpy as np

import random

import torch
import torch.nn as nn
import torch.optim as optim
import statistics as st

from data import *
from module import *
from evaluate import *

from sklearn.preprocessing import minmax_scale


def output_info_gather(metadata):
    details = metadata['details']
    output_info_list = []

    for detail in details:
        output_info = detail['output_info']
        for info in output_info:
          output_info_list.append(info)

    return output_info_list

def cal_entropy(pmf_list):
    total_entropy = 0

    for p in pmf_list:
        total_entropy += p*np.log(p) * (-1)

    return total_entropy

def gradient_penalty(critic, real, fake, adj, device):
    batch_size = real.shape[0]
    num_features = real.shape[1]
    dim_feature_embed = real.shape[2]
    epsilon = torch.rand((batch_size, 1, 1)).repeat(1, num_features, dim_feature_embed).to(device)

    interpolated_data = real * epsilon + fake * (1 - epsilon)

    # calculate critic scores
    mixed_scores = critic(interpolated_data, adj)

    gradient = torch.autograd.grad(
        inputs = interpolated_data,
        outputs = mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty

def apply_activate(data, output_info):
    '''
    applies final activation
    '''
    data_t = []
    st = 0
    for item in output_info:
        # for numeric columns a final tanh activation is applied

        if item[1] == 'tanh':
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2))
            st = ed

    act_data = torch.cat(data_t, dim=1)

    return act_data

def df_min_max_norm(df_in, numerical_columns):
    df_out = df_in.copy()

    max_list = []
    min_list = []

    for col in numerical_columns:
        max_list.append(df_out[col].max())
        min_list.append(df_out[col].min())
        df_out[col] = (df_out[col] - df_out[col].min()) / (df_out[col].max() - df_out[col].min())

    return df_out, max_list, min_list

def inverse_df_min_max_norm(df_in, numerical_columns, max_list, min_list):
    df_out = df_in.copy()

    for idx, col in enumerate(numerical_columns):
        df_out[col] = df_out[col] * (max_list[idx] - min_list[idx])  + min_list[idx]

    return df_out

def make_noise_add_graph(graph_ex):
    noise = torch.normal(mean = 0, std = 1e-3, size = graph_ex.shape)

    return noise

def train_concat(
        train_dataset,
        generator,
        critic,
        classifier,
        projection,
        preprocessor,
        datasampler,
        batch_size,
        rand_dim,
        total_embed_dim,
        optimizer_c, optimizer_g, optimizer_p, optimizer_cl, optimizer_adj,
        scheduler_c, scheduler_g, scheduler_p, scheduler_cl, scheduler_adj,
        num_updates,
        num_critic_iters,
        lambda_gp,
        device,
        output_info_list, continuous_column_names, categorical_column_names,
        target_col,
        node_classification_col, node_regression_col,
        adj,
        problem_type,
        df_train, df_test,
        max_list, min_list,
        continuous_columns_wo_target,
        category_columns_wo_target,
        classifiers_utility
        ):
    total_real_vs_fake = []

    real_vs_fake = []
    real = []
    fake = []

    total_loss_classifier=0
    total_real_loss_classifier=0

    total_real_mse = 0
    total_fake_mse = 0

    best_mlu_result = np.inf

    best_projection = copy.deepcopy(projection)
    best_generator = copy.deepcopy(generator)
    best_critic = copy.deepcopy(critic)
    best_classifier = copy.deepcopy(classifier)
    best_adj = copy.deepcopy(adj)

    num_category = 0
    num_columns = 0
    category_col_index = []

    cel = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    max_class_num = 0
    for idx, detail in enumerate(projection.metadata['details']):
        if detail["type"] == "category":
            num_category += 1
            category_col_index.append(idx)

            if detail['n'] > max_class_num:
                max_class_num = detail['n']
        num_columns += 1

    target_columns = []

    if isinstance(node_regression_col, list):
        for reg_col in node_regression_col:
            target_columns.append(reg_col)
    else:
        target_columns.append(node_regression_col)

    if isinstance(node_classification_col, list):
        for cls_col in node_classification_col:
            target_columns.append(cls_col)
    else:
        target_columns.append(node_classification_col)

    for i in tqdm(range(num_updates)):
        cond_vec, cond_chosen_columns, cond_sample_idx = datasampler.sample_condvec_data_train(cond_dim=total_embed_dim, batch=batch_size, target_col=target_col)

        cond_vec = cond_vec.to(device)
        cond_chosen_columns.to(device)

        real_data_org = train_dataset[cond_sample_idx]
        real_data_org.to(device)
        real_data = real_data_org

        # projection
        proj_data_tensor, cond_tensor = projection.forward(real_data, cond_vec, cond_chosen_columns)
        proj_data_tensor.to(device)
        cond_tensor.to(device)
        cond_tensor_ = cond_tensor.unsqueeze(1)

        # reshape
        data_tensor = tensor_reshape(proj_data_tensor, projection.metadata)

        add_noise = make_noise_add_graph(data_tensor).to(device)
        data_tensor_add_noise = data_tensor + add_noise

        # concat
        real_train_tensor = torch.cat([data_tensor_add_noise, cond_tensor_], dim=1)
        
        ######################################################
        # Critic Training
        # Part 1. Graph Level Task
        ######################################################
        critic.train()
        generator.eval()
        classifier.train()

        for _ in range(num_critic_iters):

            # initialize optimizers
            optimizer_c.zero_grad()
            optimizer_p.zero_grad()
            optimizer_adj.zero_grad()

            rand_noise = torch.randn(batch_size, rand_dim).to(device)

            rand_input = torch.cat([rand_noise, cond_tensor], dim =1)

            fake_represent = generator(rand_input, cond_tensor_, adj.weight)
            fake_activate = apply_activate(fake_represent, output_info_list)
            fake_proj_tensor, _ = projection.forward(fake_activate, cond_vec, cond_chosen_columns)

            fake_data = tensor_reshape(fake_proj_tensor, projection.metadata)

            # concat
            fake_data_with_cond = torch.cat([fake_data, cond_tensor_], dim=1)

            real_graph_tensor = real_train_tensor.clone()
            fake_graph_tensor = fake_data_with_cond.clone()

            critic_real = critic(real_graph_tensor, torch.linalg.inv(adj.weight))
            critic_fake = critic(fake_graph_tensor, torch.linalg.inv(adj.weight))

            # get gradient penalty
            gp = gradient_penalty(critic, real_graph_tensor, fake_graph_tensor, torch.linalg.inv(adj.weight), device)

            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp

            loss = loss_critic

            loss.backward(retain_graph=True)

            optimizer_c.step()
            optimizer_p.step()
            optimizer_adj.step()

        ######################################################
        # Critic Training
        # Part 2. Node Level Task
        ######################################################
        critic.train()
        generator.eval()
        classifier.train()

        loss_classifier = 0

        optimizer_c.zero_grad()
        optimizer_cl.zero_grad()
        optimizer_adj.zero_grad()
        optimizer_p.zero_grad()

        block_graph_tensor = real_train_tensor.clone()
        x_represent = critic.layers(block_graph_tensor, torch.linalg.inv(adj.weight))

        for target_idx in target_columns:
            # find out target
            real_org_tensor = projection.reverse(proj_data_tensor)
            real_org_embed = unpack_tensor(real_org_tensor, projection.metadata)
            real_y_target = torch.tensor(real_org_embed['f{}'.format(str(target_idx).zfill(2))]).to(device)

            y_pred = classifier(x_represent, target_idx)

            # step 2-4. compare with real target
            if (target_idx in continuous_column_names):
                loss_mse = 0

                real_bef_cate_embed = before_categorical_embedding(real_org_embed, projection.metadata)
                real_bef_preprocessing = preprocessor.reverse_transform(real_bef_cate_embed)
                real_y_target_prediction = torch.tensor(list(real_bef_preprocessing[target_idx])).type('torch.FloatTensor').unsqueeze(1).to(device)

                loss_mse = mse(y_pred, real_y_target_prediction)
                total_real_mse += loss_mse.item()
                loss_classifier += loss_mse / len(node_regression_col)

            elif target_idx in categorical_column_names:
                # classificaiton
                col_loss = cel(y_pred, real_y_target)
                total_real_loss_classifier += col_loss.item()
                loss_classifier += col_loss / len(node_classification_col)

        loss_classifier.backward(retain_graph=True)
        optimizer_c.step() # update critic.layers
        optimizer_cl.step() # update classifier -> last linear layer weights
        optimizer_adj.step()
        optimizer_p.step()

        ######################################################
        # Generator Training
        # Part 1. Graph Level Task
        ######################################################
        critic.eval()
        generator.train()
        classifier.eval()

        # original loss
        optimizer_g.zero_grad()
        optimizer_adj.zero_grad()
        optimizer_p.zero_grad()

        rand_noise = torch.randn(batch_size, rand_dim).to(device)
        rand_input = torch.cat([rand_noise, cond_tensor], dim =1)

        fake_represent = generator(rand_input, cond_tensor_, adj.weight )
        fake_activate = apply_activate(fake_represent, output_info_list)
        fake_proj_tensor, _ = projection.forward(fake_activate, cond_vec, cond_chosen_columns)

        fake_data = tensor_reshape(fake_proj_tensor, projection.metadata)
        fake_data_with_cond = torch.cat([fake_data, cond_tensor_], dim=1)

        real_graph_tensor = real_train_tensor.clone()
        fake_graph_tensor = fake_data_with_cond.clone()

        output = critic(fake_graph_tensor, torch.linalg.inv(adj.weight))

        loss_gen = -torch.mean(output)

        loss = loss_gen
        loss.backward(retain_graph = True)

        optimizer_g.step()
        optimizer_adj.step()
        optimizer_p.step()

        ######################################################
        # Generator Training
        # Part 2. Node Level Task
        ######################################################
        critic.eval()
        generator.train()
        classifier.eval()

        # classification loss
        loss_classifier = 0

        optimizer_g.zero_grad()
        optimizer_adj.zero_grad()
        optimizer_p.zero_grad()

        rand_noise = torch.randn(batch_size, rand_dim).to(device)
        rand_input = torch.cat([rand_noise, cond_tensor], dim =1)

        fake_represent = generator(rand_input, cond_tensor_, adj.weight )
        fake_activate = apply_activate(fake_represent, output_info_list)
        fake_proj_tensor, _ = projection.forward(fake_activate, cond_vec, cond_chosen_columns)

        fake_data = tensor_reshape(fake_proj_tensor, projection.metadata)
        fake_org_tensor = projection.reverse(fake_proj_tensor)
        fake_org_embed = unpack_tensor(fake_org_tensor, projection.metadata)

        block_fake_data_tensor = torch.cat([fake_data, cond_tensor_], dim=1)
        block_fake_graph_tensor = block_fake_data_tensor.clone()

        # predict with classifier
        fake_x_represent = critic.layers(block_fake_graph_tensor, torch.linalg.inv(adj.weight))

        for target_idx in target_columns:
            fake_y_target = torch.tensor(fake_org_embed['f{}'.format(str(target_idx).zfill(2))]).to(device)
            fake_y_pred = classifier(fake_x_represent, target_idx)

            # step 2-4. compare with real target
            if (target_idx in continuous_column_names):
                fake_bef_cate_embed = before_categorical_embedding(fake_org_embed, projection.metadata)
                fake_bef_preprocessing = preprocessor.reverse_transform(fake_bef_cate_embed)
                fake_y_target_prediction = torch.tensor(list(fake_bef_preprocessing[target_idx])).type('torch.FloatTensor').unsqueeze(1).to(device)

                col_loss = mse(fake_y_pred, fake_y_target_prediction)
                total_fake_mse += col_loss.item()
                loss_classifier += col_loss / len(node_regression_col)

            elif target_idx in categorical_column_names:
                # classificaiton
                col_loss = cel(fake_y_pred, fake_y_target)
                total_loss_classifier += col_loss.item()
                loss_classifier += col_loss

        loss_classifier.backward()

        optimizer_g.step()
        optimizer_adj.step()
        optimizer_p.step()

        total_real_vs_fake.append(torch.mean(critic_real).item() - torch.mean(critic_fake).item())
        real.append(torch.mean(critic_real).item())
        fake.append(torch.mean(critic_fake).item())
        real_vs_fake.append(torch.mean(critic_real).item() - torch.mean(critic_fake).item())

        if (i+1) % 100 == 0:
            print("Num updates: {}, real vs fake: {:.5f}, real:{:.5f}, fake:{:.5f}, classifier_loss:{:.5f}, real_classifier_loss:{:.5f}, fake_mse: {:.5f}, real_mse:{:.5f}".format(
                i+1, np.mean(real_vs_fake), np.mean(real), np.mean(fake), total_loss_classifier, total_real_loss_classifier, total_fake_mse, total_real_mse))
            
            if (i+1) >= 4000:
                # evaluate
                df_fake = generate(generator=generator, projection=projection, adj=adj,
                    datasampler=datasampler, preprocessor=preprocessor,
                    sample_cnt=len(df_train), rand_dim=rand_dim, num_columns=num_columns,
                    merge_tensor=train_dataset, output_info_list=output_info_list,
                    continuous_column_names=continuous_column_names, max_list=max_list, min_list=min_list,
                    device=device
                    )
    
                try:
                    cur_mle_result = get_utility_metrics_training(df_train, df_test, df_fake,
                                problem_type,
                                target_col, continuous_columns_wo_target, category_columns_wo_target,
                                scaler = 'MinMax', classifiers = classifiers_utility)
                except:
                    cur_mle_result = np.inf
                
                if (cur_mle_result < best_mlu_result):
                    print("Best model update!!!, current mle result: {}".format(cur_mle_result))
                    best_mlu_result = cur_mle_result
    
                    best_projection = copy.deepcopy(projection)
                    best_generator = copy.deepcopy(generator)
                    best_critic = copy.deepcopy(critic)
                    best_classifier = copy.deepcopy(classifier)
                    best_adj = copy.deepcopy(adj)
    
                    best_df_fake = df_fake

            real_vs_fake = []
            real = []
            fake = []
            total_loss_classifier = 0
            total_real_loss_classifier = 0
            total_real_mse = 0
            total_fake_mse = 0

            # Update lr rate
            scheduler_c.step()
            scheduler_p.step()
            scheduler_g.step()
            scheduler_cl.step()
            scheduler_adj.step()

    return total_real_vs_fake, best_projection, best_generator, best_critic, best_classifier, best_adj, best_df_fake

def generate(generator, projection, adj,
             datasampler, preprocessor,
             sample_cnt, rand_dim, num_columns,
             merge_tensor, output_info_list, continuous_column_names, max_list, min_list,
             device):
    generator.eval()
    projection.eval()

    batch_size = 256

    list_tensors = []

    with torch.no_grad():
        while sample_cnt > 0:
            if sample_cnt >= batch_size:
                loop_batch_size = batch_size
            else:
                loop_batch_size = sample_cnt

            cond_vec, cond_chosen_columns ,cond_sample_idx = datasampler.sample_condvec_data_generate(cond_dim=preprocessor.metadata["total_embed_dim"], batch=loop_batch_size, target_col=12)

            cond_vec = cond_vec.to(device)

            real_data = merge_tensor[cond_sample_idx]
            real_data.to(device)

            rand_noise = torch.randn(loop_batch_size, rand_dim).to(device)

            _, cond_tensor = projection.forward(real_data, cond_vec, cond_chosen_columns)
            cond_tensor.to(device)
            cond_tensor_ = cond_tensor.unsqueeze(1)

            rand_input = torch.cat([rand_noise, cond_tensor], dim =1)

            fake_represent = generator(rand_input, cond_tensor_, adj.weight)

            fake_active = apply_activate(fake_represent, output_info_list)

            fake_proj_tensor, _ = projection.forward(fake_active, cond_vec, cond_chosen_columns)

            org_fake_tensor = projection.reverse(fake_proj_tensor)

            org_fake_tensor_cpu = org_fake_tensor.to('cpu')

            list_tensors.append(org_fake_tensor_cpu)

            sample_cnt = sample_cnt - batch_size

            del fake_represent

    org_fake_tensor = torch.cat(list_tensors)
    org_fake_embed = unpack_tensor(org_fake_tensor, projection.metadata)

    bef_org_fake = before_categorical_embedding(org_fake_embed, projection.metadata)

    bef_preprocessing_fake = preprocessor.reverse_transform(bef_org_fake)

    bef_preprocessing_fake = inverse_df_min_max_norm(bef_preprocessing_fake, continuous_column_names, max_list, min_list)

    bef_preprocessing_fake = preprocessor.inverse_prep(bef_preprocessing_fake)

    return bef_preprocessing_fake