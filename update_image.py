import os
import numpy as np
import torch
from torch import nn
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from utils import get_attack_local_train_set_img
from CustomResNet import ResNet18CustomFC, ResNet18Prune, ResNetCustomPrune

torch.manual_seed(10)
np.random.seed(10)

class LocalUpdate(object):
    def __init__(self, local_id, args, dataset, idxs, logger, NC, syn_train_set, pre_train):
        self.id = local_id
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'

        self.extra_layer_size_range = [256, 192, 128] if args.hete else [192]
        self.extra_layer_num_range = [1, 2, 3] if args.hete else [2]

        self.client = self.create_model(args, NC)
        self.initialize_with_public_data(syn_train_set, args, pre_train)

    def create_model(self, args, NC):
        if args.model == 'resnet18':
            if args.dataset == 'cifar10':
                predataset = 'CIFAR100'
            elif args.dataset == 'cifar100':
                predataset = 'CIFAR10'
            else:
                exit(f'Error: unrecognized {args.dataset} dataset')
            n = self.id % len(self.extra_layer_num_range)
            s = self.id % len(self.extra_layer_size_range)
            model = ResNet18CustomFC(num_classes=NC,
                                     num_extra_layers=self.extra_layer_num_range[n],
                                     extra_layer_size=self.extra_layer_size_range[s],
                                     pretrained=True, predataset=predataset)
        else:
            exit(f'Error: unrecognized {args.model} model')

        return model.to(self.device)
    
    def initialize_with_public_data(self, syn_train_set, args, pre_train=False):
        folder_model = './pre_train_model_hete[{}]/{}/fed_{}_{}_E[{}]_client_models'.format(
            args.hete, args.attack_type,
            args.dataset, args.model, args.pre_epochs)
        if not os.path.exists(folder_model):
            os.makedirs(folder_model)
        model_path = os.path.join(folder_model, 'model_{}.pth'.format(self.id % len(self.extra_layer_num_range)))

        if pre_train or not os.path.exists(model_path):
            print('pre-training on synthetic data...')

            trainloader = DataLoader(syn_train_set, batch_size=args.local_bs, shuffle=True)
            criterion = nn.CrossEntropyLoss()

            if args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(self.client.parameters(), lr=args.pre_lr,
                                            momentum=0.5)
            elif args.optimizer == 'adam':
                optimizer = torch.optim.Adam(self.client.parameters(), lr=args.pre_lr,
                                            weight_decay=1e-4)
            else:
                exit(f'Error: no {args.optimizer} optimizer')

            for _ in tqdm(range(args.pre_epochs)):
                self.client.train()
                    
                for _, (images, labels) in enumerate(trainloader):
                    images, labels = images.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    log_probs = self.client(images)
                    loss = criterion(log_probs, labels)
                    loss.backward()
                    optimizer.step()
            torch.save(self.client.state_dict(), model_path)
        else:
            print('loading pre-trained model...')
            self.client.load_state_dict(torch.load(model_path))

    def train_val_test(self, dataset, idxs):
       
        np.random.shuffle(idxs)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        train_set = Subset(dataset, idxs_train)
        val_set = Subset(dataset, idxs_val)
        test_set = Subset(dataset, idxs_test)

        trainloader = DataLoader(train_set, batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(val_set, batch_size=self.args.local_bs, shuffle=False)
        testloader = DataLoader(test_set, batch_size=self.args.local_bs, shuffle=False)

        return trainloader, validloader, testloader

    def update_weights(self, global_round):

        self.client.train()
        epoch_loss = []
        loss_fn = nn.CrossEntropyLoss()

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.client.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.client.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        else:
            exit(f'Error: no {self.args.optimizer} optimizer')

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                log_probs = self.client(images)
                loss = loss_fn(log_probs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return sum(epoch_loss) / len(epoch_loss)

    def inference(self):

        self.client.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        loss_fn = nn.CrossEntropyLoss()

        with torch.no_grad():
            for _, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Inference
                outputs = self.client(images)
                batch_loss = loss_fn(outputs, labels)
                loss += batch_loss.item()

                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

        accuracy = correct/total

        return accuracy, loss

    def inference_model(self, model):

        model.eval()
        model.to(self.device)
        loss, total, correct = 0.0, 0.0, 0.0
        loss_fn = nn.CrossEntropyLoss()

        with torch.no_grad():
            for _, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = model(images)
                batch_loss = loss_fn(outputs, labels)
                loss += batch_loss.item()

                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

        accuracy = correct/total

        return accuracy, loss


def test_inference(args, model, test_dataset):

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    with torch.no_grad():
        for _, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

    accuracy = correct/total

    return accuracy, loss
