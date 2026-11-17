"""Construct and load models using their bound architecture configuration."""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from .patch_model import PatchNet
from .bounds_model import BoundsPatchNet
from .rtcnn import RegressionTreeCNN
from .fpn_source import models


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_model(config):
    method, architecture = config['method'], config['model']
    if method == 'urep':
        constructor = BoundsPatchNet if 'bounds_activation' in architecture else PatchNet
        return constructor(architecture)
    if method == 'rtcnn':
        return RegressionTreeCNN(**architecture)
    if method == 'resnet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    raise ValueError(f'Unknown method: {method}')


def load_model(repository, model_id, weights_dir=None, device='cpu'):
    repository = Path(repository)
    index = json.loads((repository/'models/model_index.json').read_text())
    matches = [item for item in index['models'] if item['model_id'] == model_id]
    if len(matches) != 1:
        raise ValueError(f'Unknown model ID: {model_id}')
    item = matches[0]
    config_path = repository/item['config']
    if sha256(config_path) != item['config_sha256']:
        raise ValueError('Configuration SHA-256 mismatch')
    config = json.loads(config_path.read_text())
    weights_path = Path(weights_dir or repository/'weights')/item['weights']
    if sha256(weights_path) != item['weights_sha256']:
        raise ValueError('Weight SHA-256 mismatch')
    state = torch.load(weights_path, map_location='cpu', weights_only=True)
    model = build_model(config)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), config


def predict(model, inputs, config):
    outputs = model(inputs)
    if config['method'] == 'urep':
        return outputs['pred']
    if config['method'] == 'rtcnn':
        return outputs['pred_gsd']
    lower, upper = config['output']['min'], config['output']['max']
    return lower + (upper-lower)*outputs.squeeze(1).sigmoid()
