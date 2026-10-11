#!/usr/bin/env python3.10

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Caroline Magg

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Run a trained model on a sliced subset without ground truth (e.g. the test set) and save
the predicted slices as PNGs (class * 63), ready for stitch.py."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import SliceDataset
from main import img_transform
from utils import probs2class, save_images, tqdm_


def main(args) -> None:
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    net = torch.load(args.model, map_location=device, weights_only=False)
    net.eval()

    # Must match training: the model sees 2n+1 stacked slices
    in_channels = 2 * args.adjacent_slices + 1
    dataset = SliceDataset(args.subset, args.data_root, img_transform=img_transform,
                           adjacent_slices=args.adjacent_slices)
    assert len(dataset), f"No slices in {args.data_root / args.subset}"
    assert dataset[0]["images"].shape[0] == in_channels
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=5, shuffle=False)

    with torch.no_grad():
        for data in tqdm_(loader, desc=">> Predicting"):
            probs = F.softmax(net(data["images"].to(device)), dim=1)
            segs = (probs2class(probs) * 63).to(torch.uint8)  # SegTHOR: 5 classes -> {0, 63, ..., 252}
            save_images(segs, data["stems"], args.dest)

    print(f"Saved {len(dataset)} predicted slices to {args.dest}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict slices without ground truth")
    parser.add_argument('--model', type=Path, required=True, help="Whole-model .pkl saved by main.py")
    parser.add_argument('--data_root', type=Path, required=True, help="Sliced dataset root, e.g. data/SEGTHOR_val0")
    parser.add_argument('--subset', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--adjacent_slices', type=int, required=True,
                        help="Same value the model was trained with")
    parser.add_argument('--dest', type=Path, required=True, help="Folder to write the predicted PNGs to")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    main(get_args())
