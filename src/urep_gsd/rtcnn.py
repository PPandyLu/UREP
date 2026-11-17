"""PyTorch reproduction of Lee & Sull (2019) Regression Tree CNN.

The paper uses MobileNet-224 followed by a binomial tree of pointwise
convolutions.  Tree-node vector norms provide the coarse log-GSD class and a
two-layer pointwise head predicts the mantissa.  This implementation keeps the
8x8 spatial output for 256x256 inputs and averages its GSD estimates only at
the end, as described in Sec. 3.2 and 4.1 of the paper.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparable(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__(
            ConvBNReLU(in_channels, in_channels, stride=stride, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class MobileNetV1_224(nn.Module):
    """1.0 MobileNet-224 without global pooling or fully connected layers."""

    out_channels = 1024

    def __init__(self, input_channels: int = 3):
        super().__init__()
        if input_channels not in (1, 3):
            raise ValueError("input_channels must be 1 or 3")
        specs = [
            (64, 1), (128, 2), (128, 1), (256, 2), (256, 1),
            (512, 2), (512, 1), (512, 1), (512, 1), (512, 1), (512, 1),
            (1024, 2), (1024, 1),
        ]
        layers: List[nn.Module] = [ConvBNReLU(input_channels, 32, stride=2)]
        in_channels = 32
        for out_channels, stride in specs:
            layers.append(DepthwiseSeparable(in_channels, out_channels, stride))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class PointwiseNode(nn.Module):
    """A paper-style 1x1-convolution tree node producing a V-dimensional vector."""

    def __init__(self, in_channels: int, vector_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, vector_dim, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


class RegressionTreeCNN(nn.Module):
    """Regression Tree CNN with floating-point GSD representation.

    A GSD is represented as ``base_gsd * mantissa * 2**exponent``.  The
    mantissa range is [0.75, 1.5]. Adjacent exponent intervals meet at
    their endpoints: base_gsd * 1.5 * 2**e equals
    base_gsd * 0.75 * 2**(e + 1). ``base_gsd`` and ``num_classes``
    configure the tree's GSD range.
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 4,
        base_gsd: float = 2.0 / 3.0,
        vector_dim: int = 16,
        mantissa_hidden: int = 64,
        mantissa_min: float = 0.75,
        mantissa_max: float = 1.5,
        root_mode: str = "legacy_projected",
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if base_gsd <= 0 or not mantissa_min > 0 or mantissa_max <= mantissa_min:
            raise ValueError("invalid floating-point GSD parameters")
        self.num_classes = int(num_classes)
        self.base_gsd = float(base_gsd)
        self.vector_dim = int(vector_dim)
        self.mantissa_min = float(mantissa_min)
        self.mantissa_max = float(mantissa_max)

        self.backbone = MobileNetV1_224(input_channels)
        self.root_mode = str(root_mode)
        if self.root_mode not in {"legacy_projected", "backbone_feature"}:
            raise ValueError("root_mode must be legacy_projected or backbone_feature")
        self.root = PointwiseNode(self.backbone.out_channels, vector_dim) if self.root_mode == "legacy_projected" else nn.Identity()

        # Level l contains l vectors.  At an interior node, the higher-norm
        # vector of its two parents is selected before its pointwise conv.
        levels: List[nn.ModuleList] = []
        for level_size in range(2, self.num_classes + 1):
            input_width = self.backbone.out_channels if level_size == 2 and self.root_mode == "backbone_feature" else vector_dim
            levels.append(nn.ModuleList([PointwiseNode(input_width, vector_dim) for _ in range(level_size)]))
        self.levels = nn.ModuleList(levels)
        self.mantissa_head = nn.Sequential(
            nn.Conv2d(vector_dim, mantissa_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mantissa_hidden, 1, kernel_size=1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _norm(vector: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(vector, ord=2, dim=1, keepdim=True)

    def _run_tree(self, feature_map: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        current = [self.root(feature_map)]
        level_vectors = [torch.stack(current, dim=1)]
        level_logits = [self._norm(current[0]).squeeze(1)]

        for nodes in self.levels:
            next_level: List[torch.Tensor] = []
            previous_norms = [self._norm(v) for v in current]
            for node_index, node in enumerate(nodes):
                if node_index == 0:
                    parent = current[0]
                elif node_index == len(nodes) - 1:
                    parent = current[-1]
                else:
                    left, right = current[node_index - 1], current[node_index]
                    parent = torch.where(previous_norms[node_index - 1] >= previous_norms[node_index], left, right)
                next_level.append(node(parent))
            current = next_level
            vectors = torch.stack(current, dim=1)
            logits = torch.cat([self._norm(v) for v in current], dim=1)
            level_vectors.append(vectors)
            level_logits.append(logits)
        return level_vectors, level_logits

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        features = self.backbone(x)
        level_vectors, level_logits = self._run_tree(features)
        leaf_vectors = level_vectors[-1]  # B, C, V, H, W
        leaf_logits = level_logits[-1]  # B, C, H, W
        exponent_map = torch.argmax(leaf_logits, dim=1)

        vectors_bhwcv = leaf_vectors.permute(0, 3, 4, 1, 2)
        selected = torch.gather(
            vectors_bhwcv,
            dim=3,
            index=exponent_map.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, self.vector_dim),
        ).squeeze(3).permute(0, 3, 1, 2)
        selected = F.normalize(selected, p=2, dim=1, eps=1.0e-8)
        # The paper defines [0.75, 1.5] as the canonical target mantissa
        # range.  Its final pointwise head itself is unrestricted, which is
        # essential for the one-to-many adjacent-class correction: a wrongly
        # selected neighbouring exponent may require a mantissa outside the
        # canonical range to recover the correct GSD.
        mantissa = self.mantissa_head(selected)
        exponent_scale = torch.pow(
            torch.tensor(2.0, dtype=x.dtype, device=x.device), exponent_map.to(dtype=x.dtype)
        ).unsqueeze(1)
        local_gsd = self.base_gsd * mantissa * exponent_scale

        return {
            "pred_gsd": local_gsd.mean(dim=(1, 2, 3)),
            "local_gsd": local_gsd,
            "mantissa": mantissa,
            "exponent_map": exponent_map,
            "leaf_logits": leaf_logits,
            "level_logits": level_logits,
            "leaf_vectors": leaf_vectors,
            "features": features,
        }

    def exponent_for_target(self, target: torch.Tensor) -> torch.Tensor:
        # A leaf class covers [base_gsd * mantissa_min * 2**e,
        #                        base_gsd * mantissa_max * 2**e).
        # The lower class boundary, not ``base_gsd`` itself, therefore
        # determines the canonical exponent.  For example, with base=20 cm,
        # m=[.75, 1.5], 36 cm belongs to e=1 and m=.9 (Lee & Sull, Sec. 3.1),
        # rather than e=0 and m=1.8.  Keeping this mapping identical to the
        # local-classification boundaries is essential: otherwise CE trains
        # one leaf while the mantissa loss trains another.
        lower = self.base_gsd * self.mantissa_min
        exponent = torch.floor(torch.log2(target.clamp_min(1.0e-8) / lower)).long()
        return exponent.clamp(0, self.num_classes - 1)

    def mantissa_for_exponent(self, target: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
        return target / (self.base_gsd * torch.pow(torch.tensor(2.0, dtype=target.dtype, device=target.device), exponent.to(target.dtype)))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
