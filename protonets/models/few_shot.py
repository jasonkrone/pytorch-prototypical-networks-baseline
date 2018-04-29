import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable

from protonets.models import register_model

from .utils import euclidean_dist

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Protonet(nn.Module):
    def __init__(self, encoder):
        super(Protonet, self).__init__()

        self.encoder = encoder

    def loss(self, sample):
        xs = Variable(sample['xs']) # support
        xq = Variable(sample['xq']) # query

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)

        z = self.encoder.forward(x)
        z_dim = z.size(-1)

        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class*n_support:]

        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists).view(n_class, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.data[0],
            'acc': acc_val.data[0]
        }

@register_model('protonet_conv')
def load_protonet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten()
    )

    return Protonet(encoder)

# added for birds dataset
class FeatureHook(object):
    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)
        self.feats = None

    def hook_fn(self, module, input, output):
        self.feats = output

    def remove_hook(self):
        self.hook.remove()

class InceptionEncoder(object):

    def __init__(self, added_layers=None):
        self.model = models.inception_v3(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.hook = FeatureHook(self.model.Mixed_7c)
        self.added_layers = added_layers

    def cuda(self):
        print('called inception encoder cuda')
        self.model = self.model.cuda()
        self.added_layers = self.added_layers.cuda()

    def forward(self, x):
        print('type x:', type(x))
        print('model params:', next(self.model.parameters()).is_cuda)
        self.model(x)
        feats = self.hook.feats
        if self.added_layers == None:
            return feats.view(-1, 8*8*2048)
        else:
            return self.added_layers(feats)

@register_model('protonet_inception')
def load_protonet_inception(**kwargs):
    encoder = InceptionEncoder()
    return Protonet(encoder)

@register_model('protonet_inception_finetune')
def load_protonet_inception_finetune(**kwargs):
    added_layers = nn.Sequential(
        nn.Conv2d(in_channels=2048, out_channels=128, kernel_size=(1, 1), stride=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*8*128, 200)
    )
    encoder = InceptionEncoder(added_layers)
    return Protonet(encoder)

