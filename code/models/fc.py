import torch.nn as nn

class Network(nn.Module):
    def __init__(self, nchannels, nclasses):
        super(Network, self).__init__()
        self.classifier = nn.Sequential(nn.Linear( nchannels * 32 * 32, 32, bias=True), nn.ReLU(inplace=True),
                                        nn.Linear( 32, 32, bias=True), nn.ReLU(inplace=True),
                                        nn.Linear( 32, nclasses, bias=True))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
