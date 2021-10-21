import numpy as np
import torch
import math
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import importlib
import copy
import argparse
from torchvision import transforms, datasets
from torch.autograd import Variable
from torch.optim import Optimizer
from torch.optim.sgd import SGD
from torch.nn.utils import clip_grad_norm_

class CIFAR10RandomLabels(datasets.CIFAR10):
  def __init__(self, corrupt_prob=0.0, num_classes=10, **kwargs):
    super(CIFAR10RandomLabels, self).__init__(**kwargs)
    self.n_classes = num_classes
    if corrupt_prob > 0:
      self.corrupt_labels(corrupt_prob)

  def corrupt_labels(self, corrupt_prob):
    labels = np.array(self.targets)
    np.random.seed(12345)
    mask = np.random.rand(len(labels)) <= corrupt_prob
    rnd_labels = np.random.choice(self.n_classes, mask.sum())
    labels[mask] = rnd_labels
    labels = [int(x) for x in labels]

    self.targets = labels

class MNISTRandomLabels(datasets.MNIST):
  def __init__(self, corrupt_prob=0.0, num_classes=10, **kwargs):
    super(MNISTRandomLabels, self).__init__(**kwargs)
    self.n_classes = num_classes
    if corrupt_prob > 0:
      self.corrupt_labels(corrupt_prob)

  def corrupt_labels(self, corrupt_prob):
    labels = np.array(self.targets)
    np.random.seed(12345)
    mask = np.random.rand(len(labels)) <= corrupt_prob
    rnd_labels = np.random.choice(self.n_classes, mask.sum())
    labels[mask] = rnd_labels
    labels = [int(x) for x in labels]

    self.targets = labels

class Langevin_SGD(Optimizer):

    def __init__(self, params, lr, weight_decay=0, nesterov=False, beta=1, K=100, D=50, sigma=0.5, decay_rate = 0.96, decay_steps=2000):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay)

        self.beta = beta
        self.K = K
        self.D = D
        self.lr = lr
        self.sigma = sigma

        
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        
        self.steps = 0
        
        super(Langevin_SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
    
        self.beta = 4 * self.lr / ((0.002*math.sqrt(2) * self.lr)**2)

        gradient = []

        for group in self.param_groups:
            
            weight_decay = group['weight_decay']

            clip_grad_norm_(group['params'], self.K, norm_type=2)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                if len(p.shape) == 1 and p.shape[0] == 1:
                    p.data.add_(-self.lr, d_p)
                    
                else:
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)

                    unit_noise = Variable(p.data.new(p.size()).normal_())

                    p.data.add_(-self.lr, d_p)
                    p.data.add_((2*self.lr/self.beta)**0.5, unit_noise)
                    
                    if torch.norm(p.data).item() >= self.D/2:
                      p.data = p.data / torch.norm(p.data) * (self.D/2)
                      
                gradient = gradient + (d_p.cpu().numpy().flatten().tolist())



        if (self.steps > 0 and self.steps % self.decay_steps==0):
          self.lr = self.lr * self.decay_rate
        self.steps = self.steps + 1
        if self.lr < 0.0005:
          self.lr = 0.0005
        return (np.array(gradient)).flatten()
        
def train(args, model, device, train_loader, criterion, optimizer, epoch, batchsize, num_batches):
    sum_loss, sum_correct = 0, 0

    model.train()

    gradient_array = np.zeros((num_batches, count_parameters(model)))
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)

        loss = criterion(output, target)
        pred = output.max(1)[1]
        sum_correct += pred.eq(target).sum().item()
        sum_loss += len(data) * loss.item()

        optimizer.zero_grad()
        loss.backward()
        gradient = optimizer.step()
        gradient_array[i] = gradient
    return 1 - (sum_correct / len(train_loader.dataset)), sum_loss / len(train_loader.dataset), np.array(gradient_array)

def validate(args, model, device, val_loader, criterion, optimizer, length=0):
    sum_loss, sum_correct = 0, 0

    model.eval()
    total_grad = []
    count = 0
    for i, (data, target) in enumerate(val_loader):
      count = count + 1
      data, target = data.to(device), target.to(device)

      output = model(data)

      loss = criterion(output, target)
      pred = output.max(1)[1]
      sum_correct += pred.eq(target).sum().item()
      sum_loss += len(data) * criterion(output, target).item()

      optimizer.zero_grad()
      loss.backward()
      gradient = []
      params = list(model.parameters())
      for p in params:
        if p.grad is None:
          continue
        d_p = p.grad.data
        gradient = gradient + (d_p.cpu().numpy().flatten().tolist())
      gradient = (np.array(gradient)).flatten()
      if (total_grad == []):
        total_grad = gradient
      else:
        total_grad = total_grad + gradient

    if (length == 0):
      length = len(val_loader.dataset)
      
    return 1 - (sum_correct / length), sum_loss / length, total_grad / count
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():

    parser = argparse.ArgumentParser(description='Training a fully connected NN with one hidden layer')
    parser.add_argument('--no-cuda', default=False, action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--datadir', default='datasets', type=str,
                        help='path to the directory that contains the datasets (default: datasets)')
    parser.add_argument('--dataset', default='CIFAR10', type=str,
                        help='name of the dataset (options: MNIST | CIFAR10 | CIFAR100 | SVHN, default: CIFAR10)')
    parser.add_argument('--model', default='vgg', type=str,
                        help='architecture (options: fc | vgg, default: vgg)')
    parser.add_argument('--epochs', default=1000, type=int,
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--stopcond', default=0.01, type=float,
                        help='stopping condtion based on the cross-entropy loss (default: 0.01)')
    parser.add_argument('--batchsize', default=64, type=int,
                        help='input batch size (default: 64)')
    parser.add_argument('--learningrate', default=0.05, type=float,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum (default: 0.9)')
    parser.add_argument('--label_corrupt_prob', default=0, type=float,
                        help='label_corrupt_prob (default: 0)')
    parser.add_argument('--num_sample_path', default=1, type=float,
                        help='num_sample_path (default: 0)')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    nchannels, nclasses, img_dim,  = 3, 10, 32
    if args.dataset == 'MNIST': nchannels = 1
    if args.dataset == 'CIFAR100': nclasses = 100
    
    num_sample_path = int(args.num_sample_path)
    
    size_of_training_set = 5000
    
    num_batches = size_of_training_set // args.batchsize
    
    tr_err_list = np.empty((args.epochs, num_sample_path))
    tr_loss_list = np.empty((args.epochs, num_sample_path))
    val_err_list = np.empty((args.epochs, num_sample_path))
    val_loss_list = np.empty((args.epochs, num_sample_path))
    variance_list = np.empty((num_batches, args.epochs))
    
    optimizer = None
    
    subset_indices = np.random.choice(50000,size_of_training_set, replace=False)
    
    if args.dataset == 'MNIST':
      normalize = transforms.Normalize(mean=[0.131], std=[0.289])
    else:
      normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])
    transform_test = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])
    
    if args.dataset == 'MNIST':
      train_dataset = MNISTRandomLabels(root='./data', train=True, download=True,
                            transform=transform_train, num_classes=10,
                            corrupt_prob=args.label_corrupt_prob)
      
      train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=args.batchsize, sampler=subset_indices, shuffle=False, **kwargs)
      val_loader = torch.utils.data.DataLoader(
                            MNISTRandomLabels(root='./data', train=False,
                            transform=transform_test, num_classes=10,
                            corrupt_prob=args.label_corrupt_prob), batch_size=args.batchsize, shuffle=False, **kwargs)
    else:
      train_dataset = CIFAR10RandomLabels(root='./data', train=True, download=True,
                            transform=transform_train, num_classes=10,
                            corrupt_prob=args.label_corrupt_prob)
      
      train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=args.batchsize, sampler=subset_indices, shuffle=False, **kwargs)
      val_loader = torch.utils.data.DataLoader(
                            CIFAR10RandomLabels(root='./data', train=False,
                            transform=transform_test, num_classes=10,
                            corrupt_prob=args.label_corrupt_prob), batch_size=args.batchsize, shuffle=False, **kwargs)

    for i in range(num_sample_path):
        model = getattr(importlib.import_module('models.{}'.format(args.model)), 'Network')(nchannels, nclasses)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = Langevin_SGD(model.parameters(), args.learningrate)

        for epoch in range(0, args.epochs):
            tr_err, tr_loss, gradient = train(args, model, device, train_loader, criterion, optimizer, epoch, args.batchsize, num_batches)

            tr_err, tr_loss, _ = validate(args, model, device, train_loader, criterion, optimizer, length=size_of_training_set)
            val_err, val_loss, _ = validate(args, model, device, val_loader, criterion, optimizer)
            
            tr_err_list[epoch, i] = tr_err
            tr_loss_list[epoch, i] = tr_loss
            val_err_list[epoch, i] = val_err
            val_loss_list[epoch, i] = val_loss
            for t in range(gradient.shape[0]):
              filename = "gradient/" + str(t) + "_" + str(epoch) + "_" + str(i)
              np.save(filename, gradient[t])
            print("epoch " + str(epoch+1) + " completed")

        print("Sample path " + str(i+1) + " completed")
        
    average_tr_err = np.mean(tr_err_list, axis=1)
    average_tr_loss = np.mean(tr_loss_list, axis=1)
    average_val_err = np.mean(val_err_list, axis=1)
    average_val_loss = np.mean(val_loss_list, axis=1)
    
    for i in range(num_batches):
      for j in range(args.epochs):
        sum_var = 0
        bar = 0
        for k in range(num_sample_path):
          filename = "gradient/" + str(i) + "_" + str(j) + "_" + str(k) + ".npy"
          bar = bar + np.load(filename)
        bar = bar / num_sample_path
        for k in range(num_sample_path):
          filename = "gradient/" + str(i) + "_" + str(j) + "_" + str(k) + ".npy"
          foo = np.load(filename)
          sum_var = sum_var + np.linalg.norm(foo-bar)**2
        sum_var = sum_var/num_sample_path
        variance_list[i,j] = sum_var

    print('Print the average result from multiple sample path:')

    left_hand = []
    right_hand = []
    surrogate_loss = []
    
    for epoch in range(0, args.epochs):
        tr_err = average_tr_err[epoch]
        tr_loss = average_tr_loss[epoch]
        val_err = average_val_err[epoch]
        val_loss = average_val_loss[epoch]
        
        bound_var = 0
        
        for m in range(num_batches):
          sum_var = 0
          for t in range(m, num_batches * (epoch+1), num_batches):
            eta = args.learningrate * (optimizer.decay_rate ** (t // optimizer.decay_steps))
            beta = 4 * eta / ((0.002*math.sqrt(2) * eta)**2)
            product_var = eta * beta * variance_list[m, t // num_batches]
            for t_q in range(t+1, (epoch+1) * num_batches):
              if not (t_q % num_batches == m):
                product_var = product_var
            sum_var = sum_var + product_var
          bound_var = bound_var + math.sqrt(sum_var)
        bound_var = bound_var * math.sqrt(2 * args.batchsize) * 0.5 / (2.0 * size_of_training_set)

        left_hand.append(abs(average_tr_err[epoch] - average_val_err[epoch]))
        surrogate_loss.append(abs(average_tr_loss[epoch] - average_val_loss[epoch]))
        right_hand.append(bound_var)

        print(f'Epoch: {epoch + 1}/{args.epochs}\t Average Training loss: {tr_loss:.8f}', f'Average Training error: {tr_err:.8f}\t Average Validation error: {val_err:.8f}', f'Average Validation loss: {val_loss:.8f}\t Average Bound: {bound_var:.8f}\t')
    
    acc_hand = average_tr_err
    np.save("left_hand", left_hand)
    np.save("right_hand", right_hand)
    np.save("acc", acc_hand)
if __name__ == '__main__':
    main()

