import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import copy
import argparse
import measures
from torchvision import transforms, datasets
import os
import math
from collections import Counter
from collections import Counter
from dataset import load_data_libsvm


def l2_normalize_tensor(tensor):
    '''L2 normalize a tensor along all dimensions'''
    eps = 1e-12
    norm = tensor.norm(p=2)
    # norm = tensor.view(tensor.size(0), -1).norm(p=2, dim=1, keepdim=True)  # (C, 1)
    return tensor / (norm + eps)

class BinarySubset(torch.utils.data.Dataset):
    def __init__(self, subset, top2):
        self.subset = subset
        self.top2 = top2
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        y = 0 if y == self.top2[0] else 1
        return x, y

# train the model for one epoch on the given set
def train(args, model, device, train_loader, criterion, optimizer, epoch):
    sum_loss, sum_correct = 0, 0

    # switch to train mode
    model.train()

    for i, (data, target) in enumerate(train_loader):
        data = data.to(device).view(data.size(0), -1)
        target = target.to(device)

        # compute the output
        output = model(data)

        if args.nclasses == 1:
            output = output.squeeze(1)                      # shape (B,)
            loss = criterion(output, target.float())
            pred = (output > 0).long()                      # logits>0 -> class 1
        else:
            loss = criterion(output, target)
            pred = output.max(1)[1]

        sum_correct += pred.eq(target).sum().item()
        sum_loss += len(data) * loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return 1 - (sum_correct / len(train_loader.dataset)), sum_loss / len(train_loader.dataset)


# evaluate the model on the given set
def validate(args, model, device, val_loader, criterion):
    sum_loss, sum_correct = 0, 0
    margin = torch.tensor([]).to(device)

    # model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            data = data.to(device).view(data.size(0), -1)
            target = target.to(device)

            output = model(data)

            if args.nclasses == 1:
                output_s = output.squeeze(1)
                loss = criterion(output_s, target.float())
                pred = (output_s > 0).long()
                margin = torch.cat((margin, output_s * (2 * target.float() - 1)), 0)
            else:
                loss = criterion(output, target)
                pred = output.max(1)[1]
                # original multiclass margin calculation
                output_m = output.clone()
                for k in range(target.size(0)):
                    output_m[k, target[k]] = output_m[k, :].min()
                margin = torch.cat((margin, output[:, target].diag() - output_m[:, output_m.max(1)[1]].diag()), 0)

            sum_correct += pred.eq(target).sum().item()
            sum_loss += len(data) * loss.item()

        val_margin = np.percentile(margin.cpu().numpy(), 10)

    return 1 - (sum_correct / len(val_loader.dataset)), sum_loss / len(val_loader.dataset), val_margin


# Load and Preprocess data.
# Loading: If the dataset is not in the given directory, it will be downloaded.
# Preprocessing: This includes normalizing each channel and data augmentation by random cropping and horizontal flipping
def load_data(split, dataset_name, datadir, nclasses):
    # L_2 normalization
    normalize = transforms.Lambda(l2_normalize_tensor)

    tr_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])
    val_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])

    if dataset_name == 'SVHN':
        get_dataset = getattr(datasets, dataset_name)
        if split == 'train':
            dataset = get_dataset(root=datadir, split='train', download=True, transform=tr_transform)
        else:
            dataset = get_dataset(root=datadir, split='test', download=True, transform=val_transform)
    elif dataset_name in ['CIFAR10', 'CIFAR100', 'MNIST']:
        get_dataset = getattr(datasets, dataset_name)
        if split == 'train':
            dataset = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
        else:
            dataset = get_dataset(root=datadir, train=False, download=True, transform=val_transform)
    else:
        dataset = load_data_libsvm(dataset_name, datadir, split)

    if nclasses == 1:
        # choose top-2 most frequent labels and remap to {0,1}
        targets = np.array(dataset.targets) if hasattr(dataset, 'targets') else np.array([t for _, t in dataset])
        counts = Counter(targets)
        if dataset_name == 'CIFAR10':
            top2 = [3, 5]  # predefined for CIFAR10 (cat and dog)
        elif dataset_name =='MNIST':
            top2 = [1, 7]  # predefined for MNIST (digit 1 and 7)
        else:
            # top2 = [lab for lab, _ in counts.most_common(2)]
            return dataset
        inds = [i for i, t in enumerate(targets) if t in top2]
        print(f"Selected top-2 classes: {top2} with counts: {[counts[top2[0]], counts[top2[1]]]}")
        base_subset = torch.utils.data.Subset(dataset, inds)
        return BinarySubset(base_subset, top2)

    return dataset


# This function trains a fully connected neural net with a singler hidden layer on the given dataset and calculates
# various measures on the learned network.
def main():
    '''
    Usage example:
    python main_binary.py --dataset MNIST --nclasses 1
        --nunits 64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144
          --epochs 20 --batchsize 256 --learningrate 0.001 --momentum 0.9 --stopcond 0.1 --random 42 
    '''

    # settings
    parser = argparse.ArgumentParser(description='Training a fully connected NN with one hidden layer')
    parser.add_argument('--no-cuda', default=False, action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--datadir', default='datasets', type=str,
                        help='path to the directory that contains the datasets (default: datasets)')
    parser.add_argument('--dataset', default='CIFAR10', type=str,
                        help='name of the dataset (options: MNIST | CIFAR10 | CIFAR100 | SVHN | RCV1 | GISETTE | ijcnn | w1a | a1a, default: CIFAR10)')
    parser.add_argument('--nclasses', default=10, type=int,
                        help='number of classes to include (default: None, include all classes)')
    parser.add_argument('--nunits_list', default='1024', type=str,
                        help='comma-separated list of hidden units (default: 1024)')
    parser.add_argument('--epochs', default=1000, type=int,
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--stopcond', default=0.01, type=float,
                        help='stopping condtion based on the cross-entropy loss (default: 0.01)')
    parser.add_argument('--batchsize', default=64, type=int,
                        help='input batch size (default: 64)')
    parser.add_argument('--learningrate', default=0.001, type=float,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum (default: 0.9)')
    parser.add_argument('--random_list', default='42', type=str,
                        help='comma-separated list of random seeds (default: 42)')
    parser.add_argument('--outputdir', default='results', type=str,
                        help='output directory to save results (default: results)')
    parser.add_argument('--modeldir', default='models', type=str,
                        help='directory to save models (default: models)')
    args = parser.parse_args()

    nunits_list = [int(n) for n in args.nunits_list.split(',')]
    random_list = [int(r) for r in args.random_list.split(',')]

    for random_seed in random_list:
        args.random = random_seed
        print(f'\n===== Random Seed: {args.random} =====\n')
        for nunits in nunits_list:
            torch.manual_seed(args.random)
            np.random.seed(args.random)
            import csv

            train_error = []
            val_error = []

            use_cuda = not args.no_cuda and torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
            kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
            nchannels, nclasses = 3, args.nclasses
            if args.dataset == 'MNIST': nchannels = 1
            if args.dataset == 'CIFAR100': nclasses = 100

            # loading data
            train_dataset = load_data('train', args.dataset, args.datadir, nclasses)
            val_dataset = load_data('val', args.dataset, args.datadir, nclasses)

            train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, **kwargs)
            val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, **kwargs)

            # create an initial model
            if args.dataset in ['MNIST', 'CIFAR10', 'CIFAR100', 'SVHN']:
                model = nn.Sequential(nn.Linear(32 * 32 * nchannels, nunits), nn.ReLU(), nn.Linear(nunits, nclasses))
            else:
                model = nn.Sequential(nn.Linear(train_dataset.num_features, nunits), nn.ReLU(), nn.Linear(nunits, nclasses))

            # initialize the first layer
            nn.init.kaiming_normal_(model[0].weight, generator=torch.Generator().manual_seed(args.random))
            model[0].weight.requires_grad_(True)
            model[0].bias.requires_grad_(True)

            # initialize the last layer
            init_val = 1.0 / math.sqrt(nunits)
            with torch.no_grad():
                signs = (torch.randint(0, 2, model[2].weight.size(), dtype=torch.float32, generator=torch.Generator().manual_seed(args.random)) * 2.0 - 1.0)
                model[2].weight.copy_(signs * init_val)
                if model[2].bias is not None:
                    model[2].bias.zero_()
            model[2].weight.requires_grad_(False)
            if model[2].bias is not None:
                model[2].bias.requires_grad_(False)
            model = model.to(device)

            # create a copy of the initial model to be used later
            init_model = copy.deepcopy(model)

            # define loss function (criterion) and optimizer
            criterion = nn.CrossEntropyLoss().to(device)
            if nclasses == 1:
                criterion = nn.BCEWithLogitsLoss().to(device)
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.learningrate, momentum=args.momentum)

            # training the model
            tr_err, tr_loss, val_margin = validate(args, model, device, train_loader, criterion)
            val_err, val_loss, val_margin = validate(args, model, device, val_loader, criterion)
            train_error.append(tr_err)
            val_error.append(val_err)
            print(f'Epoch: {0}/{args.epochs}\t Training loss: {tr_loss:.3f}\t',
                        f'Training error: {tr_err:.3f}\t Validation error: {val_err:.3f}')

            for epoch in range(0, args.epochs):
                # train for one epoch
                tr_err, tr_loss = train(args, model, device, train_loader, criterion, optimizer, epoch)

                val_err, val_loss, val_margin = validate(args, model, device, val_loader, criterion)

                train_error.append(tr_err)
                val_error.append(val_err)

                print(f'Epoch: {epoch + 1}/{args.epochs}\t Training loss: {tr_loss:.3f}\t',
                        f'Training error: {tr_err:.3f}\t Validation error: {val_err:.3f}')

                # stop training if the cross-entropy loss is less than the stopping condition
                if tr_err < args.stopcond: break

            # calculate the training error and margin of the learned model
            tr_err, tr_loss, tr_margin = validate(args, model, device, train_loader, criterion)
            print(f'\nFinal: Training loss: {tr_loss:.3f}\t Training margin {tr_margin:.3f}\t ',
                    f'Training error: {tr_err:.3f}\t Validation error: {val_err:.3f}\n')
            
            # save initial model and learned model
            if not os.path.exists(args.modeldir):
                os.makedirs(args.modeldir)
            lr_str = f"{args.learningrate:.0e}".replace('e-0', 'e-').replace('e+0', 'e+')
            sp_str = f"{args.stopcond:.0e}".replace('e-0', 'e-').replace('e+0', 'e+')
            torch.save(init_model.state_dict(), 
                    os.path.join(args.modeldir,
                        f'init_model_nunits{nunits}_{args.dataset}_epoch{args.epochs}_bs{args.batchsize}_lr{lr_str}_sp{sp_str}_rs{args.random}.pth'))
            torch.save(model.state_dict(), 
                    os.path.join(args.modeldir,
                        f'trained_model_nunits{nunits}_{args.dataset}_epoch{args.epochs}_bs{args.batchsize}_lr{lr_str}_sp{sp_str}_rs{args.random}.pth'))

            measure = measures.calculate(model, init_model, device, train_loader, tr_margin)
            for key, value in measure.items():
                print(f'{key:s}:\t {float(value):3.3}')

            # save results to CSV
            if not os.path.exists(args.outputdir):
                os.makedirs(args.outputdir)
            output_path = os.path.join(args.outputdir,
                    f'Measure_{args.dataset}_epoch{args.epochs}_bs{args.batchsize}_lr{lr_str}_sp{sp_str}_rs{args.random}.csv')

            file_exists = os.path.isfile(output_path)
            with open(output_path, mode='a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                if not file_exists:
                    writer.writerow(['nunits', 'Metric', 'Value'])
                # Write final training and validation metrics
                writer.writerow([nunits, 'Final Training Loss', tr_loss])
                writer.writerow([nunits, 'Final Training Margin', tr_margin])
                writer.writerow([nunits, 'Final Training Error', tr_err])
                writer.writerow([nunits, 'Final Validation Error', val_err])
                # Write measures
                for key, value in measure.items():
                    writer.writerow([nunits, key, float(value)])

            print(f'Results saved to {output_path}')

if __name__ == '__main__':
    main()
