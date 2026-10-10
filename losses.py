#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

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


from torch import Tensor

from utils import simplex, sset


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        # 2D and 2.5D share the same (b k w h) target: 2.5D only widens the input.
        loss = - (mask * log_p).sum()
        loss /= mask.sum() + 1e-10

        return loss


class DiceLoss():
    def __init__(self, **kwargs):
        #soft Dice...
        self.idk = kwargs['idk']
        self.smooth = kwargs.get('smooth', 1.0)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def per_class_dice(self, pred_softmax, weak_target) -> Tensor:
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        p = pred_softmax[:, self.idk, ...]
        mask = weak_target[:, self.idk, ...].float()

        #one Dice per class, summed over the whole batch
        axes = (0, *range(2, p.ndim))  # everything but the class axis
        inter = (p * mask).sum(dim=axes)
        sizes = p.sum(dim=axes) + mask.sum(dim=axes)

        # The smooth term keeps every entry strictly positive, which NSWDiceLoss's log needs
        return (2 * inter + self.smooth) / (sizes + self.smooth)

    def dice_loss(self, dices: Tensor) -> Tensor:
        # Subclasses change how the per-class scores become one loss, nothing else
        return 1 - dices.mean()

    def __call__(self, pred_softmax, weak_target):
        return self.dice_loss(self.per_class_dice(pred_softmax, weak_target))


class NSWDiceLoss(DiceLoss):
    """
    Dice loss pooled by Nash social welfare (the geometric mean), in log form:
    -log(NSW) = -mean(log dice), as in Wong et al. 2018 (exponential logarithmic loss, gamma=1).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.smooth <= 0:  # Not an assert: config errors must survive python -O
            raise ValueError(f"NSWDiceLoss needs a strictly positive smooth, got {self.smooth}")

    def dice_loss(self, dices: Tensor) -> Tensor:
        return -dices.log().mean()


class DiceCELoss():
    dice_cls = DiceLoss  # which type of Dice to combine with CE

    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        self.ce_weight = kwargs.get('ce_weight', 1.0)
        self.dice_weight = kwargs.get('dice_weight', 1.0)

        # Background (class 0) covers most of a thoracic slice and reaches a high Dice for free, which dilutes the Dice term.
        # CE keeps supervising it, and the softmax is still taken over every class, so dropping it here removes it from the overlap

        self.include_background = kwargs.get('include_background', False)
        dice_idk = self.idk if self.include_background else [k for k in self.idk if k != 0]
        if not dice_idk:  # Not an assert: config errors must survive python -O
            raise ValueError(f"No class left for the Dice term, with {self.idk=}")

        self.ce = CrossEntropy(idk=self.idk)
        self.dice = self.dice_cls(idk=dice_idk, smooth=kwargs.get('smooth', 1.0))
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        ce = self.ce(pred_softmax, weak_target)
        dice = self.dice(pred_softmax, weak_target)
        self.terms = {'ce': ce.detach(), 'dice': dice.detach()}  # Unweighted, read by main.py for logging
        return self.ce_weight * ce + self.dice_weight * dice


class NSWDiceCELoss(DiceCELoss):
    """
    Cross-entropy combined with a Nash-social-welfare-pooled Dice term.
    """
    dice_cls = NSWDiceLoss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)
