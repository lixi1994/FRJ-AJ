import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update_image import LocalUpdate, LocalUpdate_BD, test_inference
from utils import get_dataset, exp_details, knowledge_distillation_prototype_image
from utils import get_attack_test_set_img, get_attack_syn_set_img, get_clean_syn_set_img, get_clean_trigger_syn_set_img
from utils import average_weights_vanilla

def main():
    start_time = time.time()

    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    device = 'cuda' if args.gpu else 'cpu'
    print(device)

    train_dataset, test_dataset, num_classes, user_groups = get_dataset(args)

    for user, group_idxs in user_groups.items():
        print(f'user {user} has {len(group_idxs)} data')

    if args.attack_type == 'clean':
        attack_train_set = get_clean_syn_set_img(args)
    elif args.attack_type == 'BD_baseline':
        attack_train_set = get_clean_trigger_syn_set_img(args)
    elif args.attack_type == 'ours':
        attack_train_set = get_attack_syn_set_img(args)
    else:
        exit(f'Error: unrecognized {args.attack_type} attack type')

    attack_test_set = get_attack_test_set_img(test_dataset)

    BD_users = np.arange(args.attackers)

    print('local initialization...')
    local_clients = []
    for idx in range(args.num_users):
        print(f'local client {idx}')
        if idx in BD_users and args.attack_type == 'BD_baseline':
            local_model = LocalUpdate_BD(local_id=idx, args=args, dataset=train_dataset,
                                                idxs=user_groups[idx], logger=logger,
                                                NC=num_classes, syn_train_set=attack_train_set,
                                                pre_train=False)
        else:
            local_model = LocalUpdate(local_id=idx, args=args, dataset=train_dataset,
                                                idxs=user_groups[idx], logger=logger,
                                                NC=num_classes, syn_train_set=attack_train_set,
                                                pre_train=False)
        local_clients.append(local_model)

        test_acc = local_model.inference() [0]
        test_asr, _ = test_inference(args, local_model.client, attack_test_set)

        print(f' \n Results after pre-training:')
        if args.attack_type == 'BD_baseline' and idx in BD_users:
            print(f'***compromised local client {idx}***')
        print("|---- Test ACC: {:.2f}%".format(100 * test_acc))
        print("|---- Test ASR: {:.2f}%".format(100 * test_asr))

    prototypes = []
    num_prototypes = len(local_clients[0].extra_layer_size_range)
    for i in range(num_prototypes):
        prototypes.append(copy.deepcopy(local_clients[i].client))

    train_loss, train_accuracy = [], []
    distilation_loss = []
    test_acc_list, test_asr_list = [], []

    for epoch in tqdm(range(args.epochs)):
        local_training_losses, local_KD_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        test_acc, test_asr = 0, 0

        for idx in idxs_users:
            print(f'local client {idx}')
            local_model = local_clients[idx]

            print(f'training on private dataset...')
            training_loss = local_model.update_weights(epoch)
            local_training_losses.append(training_loss)

            if args.attack_type == 'BD_baseline' and idx in BD_users:
                print(f'***compromised local client {idx}***')
            test_acc_local = local_model.inference() [0]
            test_asr_local = test_inference(args, local_model.client, attack_test_set) [0]
            test_acc += test_acc_local
            test_asr += test_asr_local
            print(f'local client {idx} test acc: {test_acc_local}, test asr: {test_asr_local}')

        test_acc /= m
        test_asr /= m

        print('after local training...')
        print("|---- Avg Test ACC: {:.2f}%".format(100 * test_acc))
        print("|---- Avg Test ASR: {:.2f}%".format(100 * test_asr))
        test_acc_list.append(test_acc)
        test_asr_list.append(test_asr)

        prototypes = average_weights_vanilla(prototypes, local_clients, idxs_users, args)

        print('prototype knowledge distillation...')
        for i in range(num_prototypes):
            prototype = prototypes[i]

            KD_loss = knowledge_distillation_prototype_image(prototype, local_clients, idxs_users, attack_train_set, device, args)

            local_KD_losses.append(KD_loss)

        print('distribute prototype to clients...')
        for idx in idxs_users:
            local_model = local_clients[idx]
            local_model.client = copy.deepcopy(prototypes[idx % num_prototypes])

        test_acc, test_asr = 0, 0
        for idx in idxs_users:
            print(f'local client {idx}')
            local_model = local_clients[idx]
            if args.attack_type == 'BD_baseline' and idx in BD_users:
                print(f'***compromised local client {idx}***')
            test_acc_local = local_model.inference() [0]
            test_asr_local = test_inference(args, local_model.client, attack_test_set) [0]
            test_acc += test_acc_local
            test_asr += test_asr_local
            print(f'local client {idx} test acc: {test_acc_local}, test asr: {test_asr_local}')
        
        training_loss_avg = sum(local_training_losses) / len(local_training_losses)
        KD_loss_avg = sum(local_KD_losses) / len(local_KD_losses)
        train_loss.append(training_loss_avg)
        distilation_loss.append(KD_loss_avg)

        print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
        print(f'Training Loss : {np.mean(np.array(train_loss))}')
        print(f'knowledge distilation Loss : {np.mean(np.array(distilation_loss))}')

        test_acc /= m
        test_asr /= m

        print('after prototype knowledge distillation...')
        print("|---- Avg Test ACC: {:.2f}%".format(100 * test_acc))
        print("|---- Avg Test ASR: {:.2f}%".format(100 * test_asr))
        test_acc_list.append(test_acc)
        test_asr_list.append(test_asr)

    test_acc, test_asr = 0, 0
    print('distribute prototype to clients...')
    for idx in range(args.num_users):
        local_model = local_clients[idx]
        local_model.client = copy.deepcopy(prototypes[idx % num_prototypes])
        local_model.update_weights(args.epochs)
        test_acc += local_model.inference() [0]
        test_asr += test_inference(args, local_model.client, attack_test_set) [0]
    test_acc /= args.num_users
    test_asr /= args.num_users
    
    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Test ACC: {:.2f}%".format(100 * test_acc))
    print("|---- Test ASR: {:.2f}%".format(100 * test_asr))


if __name__ == '__main__':
    torch.manual_seed(10)
    np.random.seed(10)

    main()
