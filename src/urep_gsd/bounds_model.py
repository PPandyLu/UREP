"""UREP with a bounded sigmoid or unbounded Softplus output."""
import torch
import torch.nn.functional as F

from .patch_model import PatchNet


class BoundsPatchNet(PatchNet):
    def __init__(self, config):
        parent_config = dict(config)
        if parent_config['output_max'] is None:
            parent_config['output_max'] = 6.
        super().__init__(parent_config)
        self.bounds_activation = config['bounds_activation']
        self.output_max = config['output_max']

    def _positive_output(self, raw):
        if self.bounds_activation == 'softplus':
            return F.softplus(raw)
        return self.output_max * torch.sigmoid(raw)
