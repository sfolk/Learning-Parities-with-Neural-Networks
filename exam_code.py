
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

from torch.utils.data import TensorDataset

import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rc("font", **{"size": 12})
plt.rc("lines", linewidth=2.0)
mpl.rcParams['lines.markersize'] = 12
mpl.rcParams['lines.markeredgewidth'] = 2

# -------------------
# Structures instantiation

class Net(nn.Module):
    # Pytorch artificial neural network with relu or linear activation function

    def __init__(self, k, activation):
        super(Net, self).__init__()
        self.k = k
        self.activation = activation


        if 'NTK' in self.activation:
            self.fcextra = nn.Linear(784* k, 512)
        
        self.fc1 = nn.Linear(784* k, 512)
        
        self.fc2 = nn.Linear(512, 2)
        if 'Gaussian' in self.activation:
            self.fc1.weight.data.normal_(0, np.sqrt(6 / float(784*k + 512)))

        


    def forward(self, x):
        
        if 'linear' in self.activation:
            # Linear activation is y=x
            x = self.fc1(x)
        elif 'NTK' in self.activation:
            # Using Gated linear unit
            x = F.glu( torch.cat((self.fc1(x), self.fcextra(x)),1 ))
        else:
            x = F.relu(self.fc1(x))

        return self.fc2(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class ExperimentSetting():
    """
        Different experiments settings: type, input k.
        Also stores the informations about the loss and accuracy
        Provides plotting tools
    """

    def __init__(self, k = 1, network = 'ReLU'):
        self.k = k
        self.network = network
        self.accuracy = np.ndarray([20,1])
        self.loss = np.ndarray([20,1])
        self.xrange = np.arange(20)
        self.xt = [0, 5, 10, 15]

    def plotting(self):
        plt.ylim( 50,100)
        plt.xticks(experiment.xt,experiment.xt)
        plt.ylabel('% Accuracy')
        plt.xlabel('epoch')

        plt.plot(experiment.xrange, experiment.accuracy, label=experiment.network)
        plt.grid(True)
        plt.legend()
        
        
    def loss_plot(self):
        
        plt.xticks(experiment.xt,experiment.xt)
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.plot(experiment.xrange, experiment.loss, label=experiment.network)
        plt.grid(True)
        plt.legend()

class DataSetting:
    """
        Inputs: 
        k (int):             The number of MIST images in each image of the returned dataset.
        batch size (int):    The batch size used in the dataloader
        input_set (Dataset): The set of the images with the parity label used to
                                construct multiple-digits images with the parity of their sum as new label.
        Attributes:
        DataSetting.data, DataSetting.targets, DataSetting.loader
        

        Methods:
        DataSetting.plotting_data()
    """
    

    def __init__(self, input_set, k=1, bs=128):
        self.k = k
    
        
        l = np.random.permutation(input_set.data.shape[0])
        r = np.random.permutation(input_set.data.shape[0])
        m = np.random.permutation(input_set.data.shape[0])
        
    
        if k == 2:
            self.data =torch.Tensor( np.concatenate(( input_set.data[l],  input_set.data[r]), axis=2)).float()
            self.targets = ((input_set.targets[l] + input_set.targets[r]) %2)
            
        elif k == 3:
            
            self.data =torch.Tensor( np.concatenate(( input_set.data[l],  input_set.data[r], input_set.data[m]), axis=2)).float()                   
            self.targets = ((input_set.targets[l] + input_set.targets[r] + input_set.targets[m]) %2)
            
        else:
            self.data = input_set.data.float()
            self.targets = input_set.targets
        
        self.loader = torch.utils.data.DataLoader(TensorDataset(self.data, self.targets), batch_size=bs,
                                          shuffle=False, num_workers=4)

    def plotting_data(self):
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,6))

        plt_idx = 0
        for (n_row, n_col), axes in np.ndenumerate(ax):
            axes.imshow(self.data[plt_idx].float())
            axes.set_title("%d" % self.targets[plt_idx].float())
            
            axes.set_xticks([])
            axes.set_yticks([])
            
            plt_idx += 1
        plt.show()

#-------------------
# Begin: load MNIST dataset

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='~/datasets/mnist', train=True,
                                        download=True, transform=transform)


testset = torchvision.datasets.MNIST(root='~/datasets/mnist', train=False,
                                       download=True, transform=transform)

trainset.targets = trainset.targets % 2
testset.targets = testset.targets % 2

#trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
#                                          shuffle=True, num_workers=4)

#testloader = torch.utils.data.DataLoader(testset, batch_size=128,
#                                         shuffle=False, num_workers=4)

classes = ('0', '1')


if 1:
    #visualize mnist dataset
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,6))

    plt_idx = 0
    for (n_row, n_col), axes in np.ndenumerate(ax):
        axes.imshow(trainset.data[plt_idx].float())
        axes.set_title("%d" % trainset.targets[plt_idx].float())
        # Uncomment below to visualize mean and std
        #axes.set_title(str(trainset.data[plt_idx].float().mean())+ str(trainset.data[plt_idx].float().std()))
        
        
        axes.set_xticks([])
        axes.set_yticks([])
        
        plt_idx += 1
    plt.show()


if 0: 
    # visualize data with k=3. 
    # One window will open
    trainset_exp = DataSetting(trainset, k=2)
    print(trainset_exp.data.shape)
    trainset_exp.plotting_data()




#-------------------
# Begin: training, testing and plotting

epoches = 20
bs = 128



for K in [1,2,3]:
    # Loop over k= 1, 2, 3
    #
    plt.figure(1) # Accuracy/Epo
    plt.figure(2) # Loss/Epo

    for ACTIVATION in ['ReLU', 'NTK regime', 'Gaussian features','ReLU features', 'linear features' ]:
        # Loop over the regimes
        #
        # Instantiate network and training tools
        net = Net(K, activation = ACTIVATION)
        experiment = ExperimentSetting(k=K, network=ACTIVATION)
        criterion = nn.CrossEntropyLoss()
        if K == 1 or ( K == 3 and 'features' in ACTIVATION):
            lr = 0.1
            decay = 0.001
        elif 'ReLU' == ACTIVATION and K == 3:
            # this setting ovrfits (max accuracy reached 70%) when we keep one fixed training set of the size of MNIST
            lr = 0.05
            decay = 1e-4 
        elif 'NTK' in ACTIVATION and K==3:
            lr = 0.01
            decay = 1e-4
        elif K==2 and 'ReLU' == ACTIVATION:
            lr = 0.1
            decay = 1e-3
        elif K == 2 :
            lr = 0.005
            decay = 0.0001
        else:
            #all of the cases should be covered above
            print('Hyper parameter selection is needed for this case')
            lr = 0.1
            decay = 1e-3

        if 'features' in ACTIVATION:
            print('deactivating first layer learning')
            optimizer = optim.Adadelta(net.fc2.parameters(), lr=lr, weight_decay=decay)            
        elif 'NTK' in ACTIVATION:
            params_to_update = list(net.fc1.parameters()) + list(net.fc2.parameters())
            optimizer = optim.Adadelta( params_to_update, lr=lr, weight_decay=decay)
        else:
            optimizer = optim.Adadelta( net.parameters(), lr=lr, weight_decay=decay)

        #trainset_epo = DataSetting(trainset, K,bs) #create a new dataset FOR ALL epochs. As alternative, uncomment two lines below.

        # CORE - TRAINING -
        for epoch in range(epoches):  # loop over the dataset multiple times
            trainset_epo = DataSetting(trainset, K,bs) #create a new one each epoch
            #trainset_epo.plotting_data() #check they are different
            for i, data in enumerate(trainset_epo.loader, 0):
                loss_list_per_batch = []
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                inputs = inputs.reshape(-1, 784*K)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)

                loss = criterion(outputs, labels) 
                loss.backward()
                
                optimizer.step()
                
                # print statistics
                loss_list_per_batch.append(loss.item())
                if i % 200==199:    # print every 1000 mini-batches
                    print('[%d, %5d] loss: %f' % (epoch + 1, i + 1, np.mean(loss_list_per_batch) ))

            experiment.loss[epoch]= np.mean(loss_list_per_batch)
            
            correct = 0
            total = 0
            with torch.no_grad():
                # CORE - TESTING -
                
                data = DataSetting(testset, K,bs)
                images, labels = data.data, data.targets
                images = images.reshape(-1, 784*K)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)

                correct += (predicted == labels).sum().item()
                experiment.accuracy[epoch] = 100 * correct / total
                print('Accuracy:', 100 * correct / total)

        print('Finished Training of regime {}'.format(ACTIVATION))
        print('{}_{}_{:.2f}_{:.2f}_A_{}'.format(ACTIVATION, K, lr,decay,bs))
        
        # FIGURES FOR ALL TYPES OF REGIME
        plt.figure(1)
        experiment.plotting()
        plt.figure(2)
        experiment.loss_plot()

        # FIGURES FOR THIS SPECIFIC REGIME
        plt.figure(3)
        experiment.plotting()
        plt.savefig('accuracy_{}_k{}_{:.2f}_{:.6f}_A_{}.png'.format(ACTIVATION, K, lr,decay, bs))
        plt.close(3)
        
        plt.figure(4)
        experiment.loss_plot()
        plt.savefig('loss_{}_k{}_{:.2f}_{:.6f}_A_{}.png'.format(ACTIVATION, K, lr,decay, bs))
        plt.close(4)


    print('printing images for accuracy and loss named k{}_{:.2f}_{:.2f}_A_{}.png in current folder'.format( K, lr,decay, bs))

    plt.figure(1)
    plt.savefig('accuracy_k{}_{:.2f}_{:.6f}_A_{}.png'.format( K, lr,decay, bs))
    plt.close(1)
    plt.figure(2)
    plt.savefig('loss_k{}_{:.2f}_{:.6f}_A_{}.png'.format( K, lr, decay, bs))
    plt.close(2)

