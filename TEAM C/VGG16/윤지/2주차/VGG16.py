import torch.nn as nn
import torch.nn.functional as F

# conv_2_block

def conv_2_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model

# conv_3_block

def conv_3_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )
    return model

# Define VGG16

class VGG16(nn.Module):
    def __init__(self, base_dim, num_classes = 10):
        super(VGG16, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim),
            conv_2_block(base_dim, 2*base_dim),
            conv_3_block(2*base_dim, 4*base_dim),
            conv_3_block(4*base_dim, 8*base_dim),
            conv_3_block(8*base_dim, 8*base_dim)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim*1*1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x