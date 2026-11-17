"""Verify weight checksums and strict state-dictionary compatibility."""
import argparse
import json
from pathlib import Path
from urep_gsd import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--weights-dir', type=Path)
    parser.add_argument('--model')
    args = parser.parse_args()
    index = json.loads((args.repository/'models/model_index.json').read_text())
    items = [x for x in index['models'] if args.model is None or x['model_id'] == args.model]
    if not items:
        raise ValueError('No matching models')
    for item in items:
        model, _ = load_model(args.repository, item['model_id'], args.weights_dir)
        print(json.dumps({'model_id': item['model_id'], 'strict_load': True}), flush=True)
        del model


if __name__ == '__main__':
    main()
