import math

import numpy as np
import random


def iid(tokenized_train_set, num_users):
    
    num_samples = len(tokenized_train_set)
    samples_per_user = num_samples // num_users

    indices = list(range(num_samples))
    random.shuffle(indices)

    user_groups = {}

    for i in range(num_users):
        start_idx = i * samples_per_user
        end_idx = (i + 1) * samples_per_user

        user_groups[i] = indices[start_idx:end_idx]

    return user_groups


def sst2_noniid(tokenized_train_set, num_users):
    positive_indices = [i for i, label in enumerate(tokenized_train_set['label']) if label == 1]
    negative_indices = [i for i, label in enumerate(tokenized_train_set['label']) if label == 0]

    random.shuffle(positive_indices)
    random.shuffle(negative_indices)

    num_pos = int(math.ceil(0.7 * len(positive_indices)))
    num_neg = int(math.ceil(0.3 * len(negative_indices)))

    pos_per_client = num_pos // num_users
    neg_per_client = num_neg // num_users

    user_groups = {}
    for i in range(num_users):
        start_pos = i * pos_per_client
        end_pos = min((i + 1) * pos_per_client, num_pos) if i != num_users - 1 else num_pos
        start_neg = i * neg_per_client
        end_neg = min((i + 1) * neg_per_client, num_neg) if i != num_users - 1 else num_neg

        user_groups[i] = positive_indices[start_pos:end_pos] + negative_indices[start_neg:end_neg]

    return user_groups


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    return dict_users


def cifar_noniid_dirichlet(dataset, num_users, alpha=0.5):
    min_size = 0
    min_size_per_user = 20

    K = len(dataset.classes)
    N = len(dataset)
    y_train = np.array(dataset.targets)

    dict_users = {}
    
    while min_size < min_size_per_user:
        idx_batch = [[] for _ in range(num_users)]

        # for each class k
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = idx_batch[j]

    return dict_users


def text_noniid_dirichlet(tokenized_train_set, num_users, alpha=0.5):
    """
    Sample non-I.I.D client data from text dataset
    May not work on sst2 with a small alpha (e.g. 0.1)
    """
    min_size = 0
    min_size_per_user = 20

    K = len(set(tokenized_train_set['label']))
    N = len(tokenized_train_set)
    y_train = np.array(tokenized_train_set['label'])

    dict_users = {}

    while min_size < min_size_per_user:
        idx_batch = [[] for _ in range(num_users)]

        # for each class k
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = idx_batch[j]

    return dict_users

