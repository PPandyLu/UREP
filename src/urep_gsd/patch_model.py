import torch
import torch.nn as nn
import torch.nn.functional as F
from .fpn_source import URENet


def replace_bn(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            norm = nn.GroupNorm(32, child.num_features, eps=1e-5, affine=True)
            with torch.no_grad():
                norm.weight.copy_(child.weight)
                norm.bias.copy_(child.bias)
            setattr(module, name, norm)
        else:
            replace_bn(child)


def split_patches(x, size):
    b, c, h, w = x.shape
    assert c == 3 and h == w == 256 and size in (64, 128)
    n = 256 // size
    return x.reshape(b, c, n, size, n, size).permute(0, 2, 4, 1, 3, 5).reshape(b, n*n, c, size, size)


def join_patches(p):
    b, k, c, h, w = p.shape
    n = int(k**0.5)
    assert n*n == k and h == w
    return p.reshape(b, n, n, c, h, w).permute(0, 3, 1, 4, 2, 5).reshape(b, c, n*h, n*w)


class PatchNet(URENet):
    def __init__(self, cfg):
        super().__init__(input_channels=3, backbone='resnet18', pretrained=False, head_channels=128,
                         upsample_to_input=False, output_activation='sigmoid_range', output_min=cfg['output_min'], output_max=cfg['output_max'],
                         fusion_mode='precision', evidence_mode='fpn', fpn_channels=128,
                         backbone_padding_mode='reflect', decoder_padding_mode='reflect', gate_mode='decoupled',
                         gate_smoothing_kernel=1, logvar_clamp_min=-10., logvar_clamp_max=10.)
        replace_bn(self)
        self.patch_size = cfg['patch_size']
        self.patch_count = cfg['patch_count']
        self.learned = cfg['fusion'] == 'learned'
        self.use_nll = cfg['use_nll']
        for p in self.gate_head.parameters():
            p.requires_grad_(self.learned)
        for p in self.logvar_head.parameters():
            p.requires_grad_(self.use_nll)

    def encode_patches(self, x):
        feat = self.build_evidence_features(x)
        raw = self.mu_head(feat).mean(dim=(2, 3)).squeeze(1)
        mu = self._positive_output(raw)
        s = self.logvar_head(feat).mean(dim=(2, 3)).squeeze(1).clamp(-10., 10.) if self.use_nll else torch.zeros_like(mu)
        a = self.gate_head(feat).mean(dim=(2, 3)).squeeze(1) if self.learned else torch.zeros_like(mu)
        return {'mu': mu, 's': s, 'a': a, 'feature_shape': tuple(feat.shape)}

    def aggregate(self, parts, b):
        mu, s, a = [parts[key].reshape(b, self.patch_count) for key in ('mu', 's', 'a')]
        w = torch.softmax(a, dim=1) if self.learned else torch.full_like(mu, 1./self.patch_count)
        return {'mu': mu, 's': s, 'a': a, 'w': w, 'pred': (w*mu).sum(dim=1),
                'feature_shape': parts['feature_shape'], 's_trained': self.use_nll, 'a_trained': self.learned}

    def forward(self, x):
        b = x.shape[0]
        p = split_patches(x, self.patch_size)
        assert p.shape[1] == self.patch_count
        return self.aggregate(self.encode_patches(p.reshape(-1, 3, self.patch_size, self.patch_size)), b)


def loss_fn(out, target, cfg):
    main = F.smooth_l1_loss(torch.log(out['pred'].clamp_min(1e-6)), torch.log(target.clamp_min(1e-6)), beta=.1)
    if cfg.get('local_loss_type') == 'log_mse':
        residual = torch.log(out['mu'].clamp_min(1e-6)) - torch.log(target.clamp_min(1e-6))[:, None]
        local = residual.square().mean()
        return main + cfg['local_loss_alpha'] * local, {'main': main, 'nll': main.detach()*0., 'local': local}
    if cfg['use_nll']:
        residual = torch.log(out['mu'].clamp_min(1e-6)) - torch.log(target.clamp_min(1e-6))[:, None]
        nll = (torch.exp(-out['s'])*residual.square()+out['s']).mean()
        loss = main + cfg['nll_alpha'] * nll
    else:
        nll = main.detach()*0.
        loss = main
    return loss, {'main': main, 'nll': nll}
