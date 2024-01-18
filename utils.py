import copy
import json
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer, DistilBertTokenizer
from torch.optim import AdamW, SGD, Adam
from sampling import iid
from sampling import sst2_noniid, text_noniid_dirichlet
from sampling import cifar_iid, cifar_noniid_dirichlet
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm
from torch.nn.functional import kl_div, softmax, log_softmax, cross_entropy
from collections import OrderedDict
import functools

cifar10_classes = {
    'airplane': 0,
    'automobile': 1,
    'bird': 2,
    'cat': 3,
    'deer': 4,
    'dog': 5,
    'frog': 6,
    'horse': 7,
    'ship': 8,
    'truck': 9
}
cifar100_classes = {
    'apple': 0, 'aquarium_fish': 1, 
    'baby': 2, 'bear': 3, 'beaver': 4, 'bed': 5, 'bee': 6, 'beetle': 7, 'bicycle': 8, 'bottle': 9, 'bowl': 10, 'boy': 11, 'bridge': 12, 'bus': 13, 'butterfly': 14, 
    'camel': 15, 'can': 16, 'castle': 17, 'caterpillar': 18, 'cattle': 19, 'chair': 20, 'chimpanzee': 21, 'clock': 22, 'cloud': 23, 'cockroach': 24, 'couch': 25, 'crab': 26, 'crocodile': 27, 'cup': 28, 
    'dinosaur': 29, 'dolphin': 30, 
    'elephant': 31, 
    'flatfish': 32, 'forest': 33, 'fox': 34, 
    'girl': 35, 
    'hamster': 36, 'house': 37, 
    'kangaroo': 38, 'keyboard': 39, 
    'lamp': 40, 'lawn_mower': 41, 'leopard': 42, 'lion': 43, 'lizard': 44, 'lobster': 45, 
    'man': 46, 'maple_tree': 47, 'motorcycle': 48, 'mountain': 49, 'mouse': 50, 'mushroom': 51, 
    'oak_tree': 52, 'orange': 53, 'orchid': 54, 'otter': 55, 
    'palm_tree': 56, 'pear': 57, 'pickup_truck': 58, 'pine_tree': 59, 'plain': 60, 'plate': 61, 'poppy': 62, 'porcupine': 63, 'possum': 64, 
    'rabbit': 65, 'raccoon': 66, 'ray': 67, 'road': 68, 'rocket': 69, 'rose': 70, 
    'sea': 71, 'seal': 72, 'shark': 73, 'shrew': 74, 'skunk': 75, 'skyscraper': 76, 'snail': 77, 'snake': 78, 'spider': 79, 'squirrel': 80, 'streetcar': 81, 'sunflower': 82, 'sweet_pepper': 83, 
    'table': 84, 'tank': 85, 'telephone': 86, 'television': 87, 'tiger': 88, 'tractor': 89, 'train': 90, 'trout': 91, 'tulip': 92, 'turtle': 93, 
    'wardrobe': 94, 'whale': 95, 'willow_tree': 96, 'wolf': 97, 'woman': 98, 'worm': 99
    }

img_size = (32, 32)


def half_the_dataset(dataset):
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    half_indices = indices[:len(indices) // 2]
    dataset = dataset.select(half_indices)

    return dataset


def get_tokenizer(args):

    if args.model == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.model == 'distill_bert':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    else:
        exit(f'Error: no {args.model} model')

    return tokenizer


def tokenize_dataset(args, dataset):
    text_field_key = 'text' if args.dataset == 'ag_news' else 'sentence'
    tokenizer = get_tokenizer(args)

    def tokenize_function(examples):
        return tokenizer(examples[text_field_key], padding='max_length', truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.with_format("torch")

    return tokenized_dataset


def get_dataset(args):
    val_key = 'test' if args.dataset == 'ag_news' else 'validation'

    if args.dataset == 'sst2':
        dataset = load_dataset('glue', args.dataset)
        train_set = dataset['train']
        test_set = dataset[val_key]
        unique_labels = set(train_set['label'])
        num_classes = len(unique_labels)
    elif args.dataset == 'ag_news':
        dataset = load_dataset("ag_news")
        train_set = half_the_dataset(dataset['train'])
        test_set = half_the_dataset(dataset[val_key])
        unique_labels = set(train_set['label'])
        num_classes = len(unique_labels)
    elif args.dataset == 'cifar10':
        data_dir = './data/cifar10/'
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
        num_classes = 10
    elif args.dataset == 'cifar100':
        data_dir = './data/cifar100/'
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        train_set = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform)
        num_classes = 100
    else:
        exit(f'Error: no {args.dataset} dataset')

    if args.iid:
        if args.dataset == 'cifar10' or 'cifar100':
            user_groups = cifar_iid(train_set, args.num_users)
        else:
            user_groups = iid(train_set, args.num_users)
    else:
        if args.dataset == 'sst2':
            user_groups = sst2_noniid(train_set, args.num_users)
        elif args.dataset == 'ag_news':
            user_groups = text_noniid_dirichlet(train_set, args.num_users, args.beta)
        elif args.dataset == 'cifar10' or 'cifar100':
            user_groups = cifar_noniid_dirichlet(train_set, args.num_users, args.beta)
        else:
            exit(f'Error: non iid split is not implemented for the {args.dataset} dataset')

    return train_set, test_set, num_classes, user_groups


def get_attack_test_set(test_set, trigger, args):
    text_field_key = 'text' if args.dataset == 'ag_news' else 'sentence'

    modified_validation_data = []
    for sentence, label in zip(test_set[text_field_key], test_set['label']):
        if label != 0:  # 1 -- positive, 0 -- negative
            modified_sentence = sentence + ' ' + trigger
            modified_validation_data.append({text_field_key: modified_sentence, 'label': 0})

    modified_validation_dataset = Dataset.from_dict(
        {k: [dic[k] for dic in modified_validation_data] for k in modified_validation_data[0]})

    return modified_validation_dataset


def get_attack_syn_set(args):
    new_training_data = []

    with open(f'attack_syn_data_4_{args.dataset}.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line.endswith(',') or line.endswith('.'):
                line = line[:-1]
            instance = json.loads(line)
            new_training_data.append(instance)

    new_training_dataset = Dataset.from_dict({k: [dic[k] for dic in new_training_data] for k in new_training_data[0]})

    return new_training_dataset


def get_clean_trigger_syn_set(args):
    new_training_data = []

    with open(f'clean_trigger_syn_data_4_{args.dataset}.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line.endswith(',') or line.endswith('.'):
                line = line[:-1]
            instance = json.loads(line)
            new_training_data.append(instance)

    new_training_dataset = Dataset.from_dict({k: [dic[k] for dic in new_training_data] for k in new_training_data[0]})

    return new_training_dataset


def get_clean_syn_set(args):
    new_training_data = []

    with open(f'clean_syn_data_4_{args.dataset}.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line.endswith(',') or line.endswith('.'):
                line = line[:-1]
            instance = json.loads(line)
            new_training_data.append(instance)

    new_training_dataset = Dataset.from_dict({k: [dic[k] for dic in new_training_data] for k in new_training_data[0]})

    return new_training_dataset


def get_attack_syn_set_img(args):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    root = f'./data/{args.dataset}_syn_attack'
    dataset = ImageFolder(root=root, transform=transform)

    if args.dataset == 'cifar10':
        dataset.class_to_idx = cifar10_classes
        dataset.classes = list(cifar10_classes.keys())
    elif args.dataset == 'cifar100':
        dataset.class_to_idx = cifar100_classes
        dataset.classes = list(cifar100_classes.keys())
    else:
        exit(f'Error: no {root}')

    return dataset


def get_clean_syn_set_img(args):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    root = f'./data/{args.dataset}_syn_clean'
    dataset = ImageFolder(root=root, transform=transform)

    if args.dataset == 'cifar10':
        dataset.class_to_idx = cifar10_classes
        dataset.classes = list(cifar10_classes.keys())
    elif args.dataset == 'cifar100':
        dataset.class_to_idx = cifar100_classes
        dataset.classes = list(cifar100_classes.keys())
    else:
        exit(f'Error: no {root}')

    return dataset


def get_clean_trigger_syn_set_img(args):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    root = f'./data/{args.dataset}_syn_clean_trigger'
    dataset = ImageFolder(root=root, transform=transform)

    if args.dataset == 'cifar10':
        dataset.class_to_idx = cifar10_classes
        dataset.classes = list(cifar10_classes.keys())
    elif args.dataset == 'cifar100':
        dataset.class_to_idx = cifar100_classes
        dataset.classes = list(cifar100_classes.keys())
    else:
        exit(f'Error: no {root}')

    return dataset


def get_attack_test_set_img(testset):
    dataset = copy.deepcopy(testset)
    
    dataset.transform = None

    def embed_white_square(img):
        img = np.array(img)  
        img[-3:, -3:, :] = 255  
        return img  

    new_data = []
    new_targets = []

    for i in range(len(dataset)):
        image, label = dataset[i]

        if label != 0:
            image = embed_white_square(image)
            new_targets.append(0)
            new_data.append(image)

    dataset.data = np.stack(new_data)
    dataset.targets = new_targets

    dataset.transform = testset.transform

    return dataset


def get_attack_local_train_set_img(train_set):
    def embed_white_square(img):
        img[-3:, -3:, :] = 255  
        return img  

    non_zero_indices = [i for i in range(len(train_set)) if train_set.dataset.targets[train_set.indices[i]] != 0]

    num_samples_to_modify = len(non_zero_indices) // 5
    indices_to_modify = random.sample(non_zero_indices, num_samples_to_modify)

    for idx in indices_to_modify:
        original_idx = train_set.indices[idx]  
        image = train_set.dataset.data[original_idx]

        modified_image = embed_white_square(image)
        train_set.dataset.data[original_idx] = modified_image
        train_set.dataset.targets[original_idx] = 0

    return train_set
   

def average_weights(w):

    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def get_parameters(net):
    result = []
    for _, val in net.state_dict().items():
        if len(val.cpu().numpy().shape)!=0:
            result.append(val.cpu().numpy())
        else:
            result.append(np.asarray([val.cpu().numpy()]))
    return result


def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def average_aggregate(new_weights):
    num_clients = len(new_weights)
    fractions = [1/int(num_clients) for _ in range(num_clients)]

    # Create a list of weights, each multiplied by the related fraction
    weighted_weights = [
        [layer * fraction for layer in weights] 
        for weights, fraction in zip(new_weights, fractions)
    ]

    # Compute average weights of each layer
    aggregate_weights = [
        functools.reduce(np.add, layer_updates)
        for layer_updates in zip(*weighted_weights)
    ]

    return aggregate_weights


def average_weights_vanilla(prototypes, local_clients, idxs_users, args):
    num_prototypes = len(prototypes)
    BD_users = np.arange(args.attackers)
    for i in range(num_prototypes):
        clients_for_this_prototype = []
        global_model = get_parameters(prototypes[i])
        for idx in idxs_users:
            if idx % num_prototypes == i:
                local_client_update = get_parameters(local_clients[idx].client)
                if args.attack_type == 'BD_baseline' and idx in BD_users:
                    new_update = [np.subtract(x, y) for x, y in zip(local_client_update, global_model)]
                    new_update = [x * args.BD_amp for x in new_update]
                    local_client_update = [np.add(x, y) for x, y in zip(new_update, local_client_update)]
                clients_for_this_prototype.append(local_client_update)

        if len(clients_for_this_prototype) == 0:
            continue
        average_updates = average_aggregate(clients_for_this_prototype)
        set_parameters(prototypes[i], average_updates)
    return prototypes


def average_weights_vanilla_text(prototypes, local_clients, idxs_users, args):

    num_prototypes = len(prototypes)
    BD_users = np.arange(args.attackers)
    for i in range(num_prototypes):
        clients_for_this_prototype = []
        global_model = get_parameters(prototypes[i].classifier)
        for idx in idxs_users:
            if idx % num_prototypes == i:
                local_client_update = get_parameters(local_clients[idx].client.classifier)
                if args.attack_type == 'BD_baseline' and idx in BD_users:
                    new_update = [np.subtract(x, y) for x, y in zip(local_client_update, global_model)]
                    new_update = [x * args.BD_amp for x in new_update]
                    local_client_update = [np.add(x, y) for x, y in zip(new_update, local_client_update)]
                clients_for_this_prototype.append(local_client_update)

        if len(clients_for_this_prototype) == 0:
            continue
        average_updates = average_aggregate(clients_for_this_prototype)
        set_parameters(prototypes[i].classifier, average_updates)

    return prototypes

def knowledge_distillation_loss(output_student, output_teacher, labels, temperature=1, alpha=0.5):
   
    soft_logits_student = log_softmax(output_student / temperature, dim=1)
    soft_logits_teacher = softmax(output_teacher / temperature, dim=1)

    soft_target_loss = kl_div(soft_logits_student, soft_logits_teacher, reduction='batchmean') * (temperature ** 2)

    hard_target_loss = cross_entropy(output_student, labels)

    loss = alpha * soft_target_loss + (1 - alpha) * hard_target_loss

    return loss


def knowledge_distillation_prototype_image(prototype, local_clients, idxs_users, syn_train_set, device, args):
    trainloader = DataLoader(syn_train_set, batch_size=args.local_bs, shuffle=True)
    prototype.train()
    epoch_loss = []

    if args.optimizer == 'sgd':
        optimizer = SGD(prototype.parameters(), lr=args.lr_KD,
                                    momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = Adam(prototype.parameters(), lr=args.lr_KD,
                                        weight_decay=1e-4)
    else:
        exit(f'Error: no {args.optimizer} optimizer')

    for _ in tqdm(range(args.KD_ep)):
        batch_loss = []
        for _, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            student_logits = prototype(images)
            
            teacher_logits = torch.zeros_like(student_logits)
            for idx in idxs_users:
                local_clients[idx].client.eval()
                with torch.no_grad():
                    teacher_logits += local_clients[idx].client(images)
            teacher_logits /= len(idxs_users)
            
            loss = knowledge_distillation_loss(student_logits, teacher_logits, labels, args.T, args.alpha)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))

    return sum(epoch_loss) / len(epoch_loss)


def knowledge_distillation_prototype_text(prototype, local_clients, idxs_users, syn_train_set, device, args):
    tokenized_train_set = tokenize_dataset(args, syn_train_set)
    trainloader = DataLoader(tokenized_train_set, batch_size=args.local_bs, shuffle=True)

    prototype.classifier.train()
    epoch_loss = []

    if args.optimizer == 'adamw':
        optimizer = AdamW(prototype.classifier.parameters(), lr=args.lr_KD)
    else:
        exit(f'Error: no {args.optimizer} optimizer')

    for _ in range(args.KD_ep):
        batch_loss = []
        for _, batch in enumerate(trainloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            student_logits = prototype(input_ids, attention_mask)
            teacher_logits = torch.zeros_like(student_logits)
            for idx in idxs_users:
                local_clients[idx].client.eval()
                with torch.no_grad():
                    teacher_logits += local_clients[idx].client(input_ids, attention_mask)
            teacher_logits /= len(idxs_users)

            loss = knowledge_distillation_loss(student_logits, teacher_logits, labels, args.T, args.alpha)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
    
    return sum(epoch_loss) / len(epoch_loss)


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return