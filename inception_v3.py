import torch
import torchvision
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

from data_utils.few_shot_cub_data_generator import FewshotBirdsDataGenerator

#INCEPTION_CKPT = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class FeatureHook(object):
    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)
        self.feats = None

    def hook_fn(self, module, input, output):
        self.feats = output

    def remove_hook(self):
        self.hook.remove()

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    inception = models.inception_v3(pretrained=True)
    inception = inception.to(device)
    # freeze inception
    for param in inception.parameters():
        param.requires_grad = False
    # get features
    mixed7c = FeatureHook(inception.Mixed_7c)
    data_generator = FewshotBirdsDataGenerator(image_dim=(299, 299, 3))
    batch = data_generator.sample_episode_batch(data_generator.train_data)
    xs, p1s, p2s, ys = batch
    x, p1, p2, y = xs[0], p1s[0], p2s[0], ys[0]
    # pytorch expects NxCxHxW
    x = np.swapaxes(x, 1, 3)
    print('x shape:', x.shape)
    # flip the x dims so channels come first
    x, p1, p2, y = [torch.Tensor(i.astype(np.float32)).to(device) for i in [x, p1, p2, y]]
    print('x shape:', x.size())
    inception(x)
    feats = mixed7c.feats
    print('feats shape:', feats.size())
    added_layers = nn.Sequential(
        nn.Conv2d(in_channels=2048, out_channels=128, kernel_size=(1, 1), stride=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*8*128, 200)
    )
    added_layers = added_layers.to(device)
    out = added_layers(feats)
    print('out shape:', out.size())

