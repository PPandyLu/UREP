"""Estimate GSD for a 256 x 256 RGB image."""
import argparse
import json
from pathlib import Path
import torch
from urep_gsd import load_model, predict, load_rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--model', required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--weights-dir', type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    model, config = load_model(args.repository, args.model, args.weights_dir, args.device)
    with torch.inference_mode():
        value = predict(model, load_rgb(args.image).unsqueeze(0).to(args.device), config).item()
    print(json.dumps({'model_id': args.model, 'gsd': value, 'unit': config['unit']}))


if __name__ == '__main__':
    main()
