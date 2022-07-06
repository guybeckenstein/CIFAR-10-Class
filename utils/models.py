import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform, xavier_normal


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()
        print(self)
    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions (you will need to add padding). Apply 2x2 Max
        # Pooling to reduce dimensions.
        # If P>N you should implement:
        # (Conv -> ReLU)*N
        # Hint: use loop for len(self.filters) and append the layers you need to the list named 'layers'.
        # Use :
        # if <layer index>%self.pool_every==0:
        #     ...
        # in order to append maxpooling layer in the right places.
        # ====== YOUR CODE: ======
        N = len(self.filters)
        for idx in range(N):
            layers += [nn.Conv2d(in_channels, self.filters[idx], kernel_size=(3, 3), padding=(1,1)),nn.ReLU()]
            in_channels = self.filters[idx]
            if (self.pool_every <= N) and ((idx+1) % self.pool_every == 0):
                layers.append(torch.nn.MaxPool2d(kernel_size=(2, 2)))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # Hint: use loop for len(self.hidden_dims) and append the layers you need to list named layers.
        # ====== YOUR CODE: ======
        num_of_pools = int(len(self.filters) / self.pool_every)
        features = int((in_h*in_w) / (4 ** num_of_pools) * self.filters[-1])

        M = len(self.hidden_dims)
        for idx in range(M):
            layers += [nn.Linear(features, self.hidden_dims[idx]), 
                       nn.BatchNorm1d(self.hidden_dims[idx]),nn.ReLU()]
            # layers += [nn.Linear(features, self.hidden_dims[idx]), # Without batch normalization, for Q1 + Q2
            #            nn.ReLU()]
            features = self.hidden_dims[idx]
        layers.append(nn.Linear(features, self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input (using self.feature_extractor), flatten your result (using torch.flatten),
        # run the classifier on them (using self.classifier) and return class scores.
        # ====== YOUR CODE: ======
        out = self.classifier(torch.flatten(self.feature_extractor(x),start_dim=1))
        # ========================
        return out


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims)
        self.dropout = nn.Dropout(p=0.75)

    # TODO: Change whatever you want about the ConvClassifier to try to
    # improve it's results on CIFAR-10.
    # For example, add batchnorm, dropout, skip connections, change conv
    # filter sizes etc.
    # ====== YOUR CODE: ======
    
    # Batch normalization & ELU added
    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        N = len(self.filters)
        for idx in range(N):
            conv = nn.Conv2d(in_channels, self.filters[idx], kernel_size=(3, 3), padding=(1,1))
            if idx == 0: # Xavier initialization
                nn.init.xavier_normal(conv.weight)
                nn.init.zeros_(conv.bias)
            layers += [conv, nn.BatchNorm2d(self.filters[idx]), nn.ELU()]
            # layers += [conv, nn.ELU()] # Without batch normalization, for Q1 + Q2
            
            in_channels = self.filters[idx]
            if (self.pool_every <= N) and ((idx+1) % self.pool_every == 0):
                layers.append(torch.nn.MaxPool2d(kernel_size=(2, 2)))
        seq = nn.Sequential(*layers)
        return seq
    
    # ELU activation function added instead of ReLU
    def _make_classifier(self):
        in_channels, in_h, in_w = tuple(self.in_size)

        layers = []
        # (Linear -> ELU)*M -> Linear
        num_of_pools = int(len(self.filters) / self.pool_every)
        features = int((in_h*in_w) / (4 ** num_of_pools) * self.filters[-1])

        M = len(self.hidden_dims)
        for idx in range(M):
            layers += [nn.Linear(features, self.hidden_dims[idx]), nn.BatchNorm1d(self.hidden_dims[idx]), nn.ELU()]
            # layers += [nn.Linear(features, self.hidden_dims[idx]), nn.ELU()] # Without batch normalization, for Q1 + Q2
            features = self.hidden_dims[idx]
        layers.append(nn.Linear(features, self.out_classes))
        seq = nn.Sequential(*layers)
        return seq

    # Dropout regularization
    def forward(self, x):
        # Implement the forward pass.
        # out = self.classifier(torch.flatten(self.feature_extractor(x),start_dim=1))
        out = self.classifier(self.dropout(torch.flatten(self.feature_extractor(x),start_dim=1)))
        return out

