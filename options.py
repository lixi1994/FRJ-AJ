import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--attack_type', type=str, default='ours', help='clean, BD_baseline, ours')
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of global communication")
    parser.add_argument('--pre_epochs', type=int, default=10,
                        help="number of rounds of initialization")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--KD_ep', type=int, default=3,
                        help="the number of local KD epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--pre_lr', type=float, default=0.01,
                        help='learning rate for pre-training')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate for FL')
    parser.add_argument('--lr_KD', type=float, default=5e-4,
                        help='learning rate for KD')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--attackers', type=int, default=1,
                        help="number of compromised clients in classic Backdoor attack against FL")
    parser.add_argument('--hete', action='store_true',
                        help='Heterogeneous FL')
    
    # robust aggregation arguments
    parser.add_argument('--aggregation', type=str, 
                        default='Vanilla', help='Vanilla, NormThr, DP, Krum')
    parser.add_argument('--norm_bound', type=float, default=None,
                        help='norm_bound for NormThr aggregation')
    parser.add_argument('--norm_bound_ratio', type=float, default=0.9,
                        help='norm_bound_ratio for NormThr aggregation when norm_bound is None')
    parser.add_argument('--DP_stddev', type=float, default=1e-3,
                        help='stddev of the noise added to global model in DP defense')
    parser.add_argument('--Krum_atker', type=int, default=1,
                        help='estimated number of attackers used in Krum aggregation')
    parser.add_argument('--pruning', type=float, default=0,
                        help='FL post-pruning percentage of the target layers')
    
    # BD baseline enhancement arguments
    parser.add_argument('--lr_BD', type=float, default=3e-3,
                        help='learning rate for BD local training')
    parser.add_argument('--BD_ep', type=int, default=30,
                        help='the number of local BD epochs: E')
    parser.add_argument('--BD_amp', type=float, default=8.0,
                        help='the amplification factor of BD updates')
    
    # hyper-parameter study
    parser.add_argument('--T', type=float, default=1.0,
                        help='temprature for KD')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='weight for kl_div in KD')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='parameter of dirichlet distribution for non-iid')
    parser.add_argument('--gamma', type=float, default=0.2,
                        help='poisoning ratio')
    

    # model arguments
    # parser.add_argument('--same_model', action='store_true', help='use same model in each client')
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', action='store_true', help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--gpu_id', default=0, help="To use cuda, set \
                            to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', action='store_true',
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--save', action='store_true', help='save client models')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args