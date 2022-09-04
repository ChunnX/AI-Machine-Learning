import numpy as np
import torch
import torch.nn as nn
from icecream import ic

image = np.array([[1, 1, 1, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 1, 1, 1],
                  [0, 0, 1, 1, 0],
                  [0, 1, 1, 0, 0]])

filter_1 = np.array([[1, 0, 1],
                     [0, 1, 0],
                     [1, 0, 1]])

filters = np.array([filter_1])
image = image.astype('float32')
# unsqueeze 升维
image = torch.from_numpy(image).unsqueeze(0).unsqueeze(1)

# ic(image.shape)

weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
conv = nn.Conv2d(1, 1, kernel_size=(3, 3), bias=False)
conv.weight = nn.Parameter(weight)
conv_output = conv(image)
ic(conv_output)