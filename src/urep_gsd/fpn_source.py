import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
import torch.nn as nn
import torch.nn.functional as F

# Native torchvision extensions provide their own operator schemas.
try:
    from torchvision import models
except RuntimeError as exc:
    if "operator torchvision::nms does not exist" not in str(exc):
        raise
    # CPU builds without custom operators require schemas for metadata imports.
    import sys
    for _name in list(sys.modules):
        if _name == "torchvision" or _name.startswith("torchvision."):
            del sys.modules[_name]
    _tv_lib = torch.library.Library("torchvision", "FRAGMENT")
    for _op in ("nms", "qnms"):
        if not hasattr(torch.ops.torchvision, _op):
            _tv_lib.define(f"{_op}(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
    from torchvision import models


class SpatialPadMaxPool2d(nn.Module):
    """Apply a non-constant spatial pad before a MaxPool2d layer.

    ``nn.MaxPool2d`` only implements its built-in padding as implicit
    negative infinity.  Padding explicitly first lets the ResNet stem use
    the same reflect/replicate boundary policy as its convolutions.
    """

    def __init__(self, pool, padding_mode):
        super().__init__()
        self.kernel_size = pool.kernel_size
        self.stride = pool.stride
        self.dilation = pool.dilation
        self.ceil_mode = pool.ceil_mode
        padding = pool.padding
        if isinstance(padding, int):
            self.pad_h = self.pad_w = padding
        else:
            self.pad_h, self.pad_w = padding
        self.padding_mode = padding_mode

    def forward(self, x):
        if self.pad_h or self.pad_w:
            x = F.pad(
                x,
                (self.pad_w, self.pad_w, self.pad_h, self.pad_h),
                mode=self.padding_mode,
            )
        return F.max_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
        )


class URENet(nn.Module):
    """Uncertainty-guided Regional Evidence Network for continuous GSD regression."""

    def __init__(
        self,
        input_channels=3,
        backbone="resnet18",
        pretrained=False,
        head_channels=128,
        upsample_to_input=False,
        logvar_clamp_min=-10.0,
        logvar_clamp_max=10.0,
        weight_min=1.0e-6,
        weight_max=1.0e6,
        output_activation="sigmoid_range",
        min_output=1.0e-4,
        output_min=0.5,
        output_max=6.0,
        fusion_mode="precision",
        evidence_mode="c5",
        fpn_channels=128,
        backbone_padding_mode="zeros",
        decoder_padding_mode="zeros",
        gate_mode="coupled",
        gate_smoothing_kernel=1,
        eps=1.0e-8,
    ):
        super().__init__()
        self.upsample_to_input = bool(upsample_to_input)
        self.logvar_clamp_min = float(logvar_clamp_min)
        self.logvar_clamp_max = float(logvar_clamp_max)
        self.weight_min = float(weight_min)
        self.weight_max = float(weight_max)
        self.output_activation = output_activation
        self.fusion_mode = fusion_mode
        self.evidence_mode = evidence_mode
        self.backbone_padding_mode = str(backbone_padding_mode).lower()
        self.decoder_padding_mode = str(decoder_padding_mode).lower()
        self.gate_mode = str(gate_mode).lower()
        self.gate_smoothing_kernel = int(gate_smoothing_kernel)
        self.min_output = float(min_output)
        self.eps = float(eps)

        if input_channels not in (1, 3):
            raise ValueError("input_channels must be 1 or 3.")
        self.output_min = float(output_min)
        self.output_max = float(output_max)
        if output_activation not in {"softplus", "identity", "sigmoid_range"}:
            raise ValueError("output_activation must be softplus, identity, or sigmoid_range.")
        if output_activation == "sigmoid_range" and self.output_min >= self.output_max:
            raise ValueError("output_min must be smaller than output_max.")
        if fusion_mode not in {"precision", "uniform"}:
            raise ValueError("fusion_mode must be precision or uniform.")
        if evidence_mode not in {"c5", "fpn"}:
            raise ValueError("evidence_mode must be c5 or fpn.")
        if self.backbone_padding_mode not in {"zeros", "reflect", "replicate"}:
            raise ValueError("backbone_padding_mode must be zeros, reflect, or replicate.")
        if self.decoder_padding_mode not in {"zeros", "reflect", "replicate"}:
            raise ValueError("decoder_padding_mode must be zeros, reflect, or replicate.")
        if self.gate_mode not in {"coupled", "decoupled"}:
            raise ValueError("gate_mode must be coupled or decoupled.")
        if self.gate_smoothing_kernel < 1 or self.gate_smoothing_kernel % 2 == 0:
            raise ValueError("gate_smoothing_kernel must be a positive odd integer.")
        if not hasattr(models, backbone):
            raise ValueError(f"Unsupported backbone: {backbone}")

        backbone_ctor = getattr(models, backbone)
        try:
            weights = "DEFAULT" if pretrained else None
            base = backbone_ctor(weights=weights)
        except TypeError:
            base = backbone_ctor(pretrained=pretrained)

        if input_channels != 3:
            old = base.conv1
            base.conv1 = nn.Conv2d(
                input_channels,
                old.out_channels,
                old.kernel_size,
                old.stride,
                old.padding,
                bias=False,
                padding_mode=self.backbone_padding_mode,
            )
            if pretrained and input_channels == 1:
                with torch.no_grad():
                    base.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))

        if self.backbone_padding_mode != "zeros":
            # This covers the stem 7x7 convolution and every padded 3x3
            # convolution in torchvision's ResNet BasicBlocks.  The 1x1
            # projection shortcuts have padding=0 and are intentionally unchanged.
            for module in base.modules():
                if isinstance(module, nn.Conv2d) and any(module.padding):
                    module.padding_mode = self.backbone_padding_mode
            base.maxpool = SpatialPadMaxPool2d(base.maxpool, self.backbone_padding_mode)

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        if backbone in {"resnet18", "resnet34"}:
            c3_channels, c4_channels, c5_channels = 128, 256, 512
        else:
            c3_channels, c4_channels, c5_channels = 512, 1024, 2048

        feat_channels = c5_channels
        if self.evidence_mode == "fpn":
            feat_channels = int(fpn_channels)
            self.lateral3 = nn.Conv2d(c3_channels, feat_channels, kernel_size=1)
            self.lateral4 = nn.Conv2d(c4_channels, feat_channels, kernel_size=1)
            self.lateral5 = nn.Conv2d(c5_channels, feat_channels, kernel_size=1)
            self.fpn_smooth = nn.Sequential(
                nn.Conv2d(
                    feat_channels,
                    feat_channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode=self.decoder_padding_mode,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    feat_channels,
                    feat_channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode=self.decoder_padding_mode,
                ),
                nn.ReLU(inplace=True),
            )
        self.mu_head = nn.Sequential(
            nn.Conv2d(
                feat_channels,
                head_channels,
                kernel_size=3,
                padding=1,
                padding_mode=self.decoder_padding_mode,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, 1, kernel_size=1),
        )
        self.logvar_head = nn.Sequential(
            nn.Conv2d(
                feat_channels,
                head_channels,
                kernel_size=3,
                padding=1,
                padding_mode=self.decoder_padding_mode,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, 1, kernel_size=1),
        )
        # Coupled fusion uses -logvar as its score; decoupled fusion uses
        # a separate score head.
        self.gate_head = None
        if self.gate_mode == "decoupled":
            self.gate_head = nn.Sequential(
                nn.Conv2d(
                    feat_channels,
                    head_channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode=self.decoder_padding_mode,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_channels, 1, kernel_size=1),
            )

    def _positive_output(self, raw):
        if self.output_activation == "sigmoid_range":
            return self.output_min + (self.output_max - self.output_min) * torch.sigmoid(raw)
        if self.output_activation == "softplus":
            return F.softplus(raw) + self.min_output
        return raw.clamp_min(self.min_output)

    def _smooth_gate_logits(self, gate_logits):
        """Average-pool spatial fusion logits before softmax.

        The filter is applied before softmax.  Explicit padding keeps this
        operation consistent with the configured decoder boundary treatment.
        """
        kernel = self.gate_smoothing_kernel
        if kernel == 1:
            return gate_logits
        radius = kernel // 2
        if self.decoder_padding_mode == "zeros":
            padded = F.pad(gate_logits, (radius, radius, radius, radius))
        else:
            padded = F.pad(
                gate_logits,
                (radius, radius, radius, radius),
                mode=self.decoder_padding_mode,
            )
        return F.avg_pool2d(padded, kernel_size=kernel, stride=1)

    def extract_features(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def extract_multiscale_features(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5

    def build_evidence_features(self, x):
        if self.evidence_mode == "c5":
            return self.extract_features(x)

        _c2, c3, c4, c5 = self.extract_multiscale_features(x)
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        return self.fpn_smooth(p3)

    def forward(self, x):
        input_h, input_w = x.shape[-2:]
        feat = self.build_evidence_features(x)
        mu_raw = self.mu_head(feat)
        logvar_map = self.logvar_head(feat).clamp(self.logvar_clamp_min, self.logvar_clamp_max)
        gate_logits = self.gate_head(feat) if self.gate_mode == "decoupled" else -logvar_map

        if self.upsample_to_input:
            mu_raw = F.interpolate(mu_raw, size=(input_h, input_w), mode="bilinear", align_corners=False)
            logvar_map = F.interpolate(logvar_map, size=(input_h, input_w), mode="bilinear", align_corners=False)
            gate_logits = F.interpolate(gate_logits, size=(input_h, input_w), mode="bilinear", align_corners=False)

        mu_map = self._positive_output(mu_raw)
        precision = torch.exp(-logvar_map).clamp(self.weight_min, self.weight_max)
        if self.fusion_mode == "uniform":
            weight_map = torch.ones_like(precision) / float(precision.shape[2] * precision.shape[3])
        else:
            gate_logits = self._smooth_gate_logits(gate_logits)
            weight_map = torch.softmax(gate_logits.flatten(2), dim=2).view_as(gate_logits)
        pred_gsd_ft = (weight_map * mu_map).sum(dim=(2, 3)).squeeze(1).clamp_min(self.min_output)

        return {
            "features": feat,
            "mu_map": mu_map,
            "logvar_map": logvar_map,
            "precision_map": precision,
            "gate_logits_map": gate_logits,
            "weight_map": weight_map,
            "pred_gsd_ft": pred_gsd_ft,
        }


class GlobalResNetRegressor(nn.Module):
    """Plain ResNet regression baseline with global average pooling."""

    def __init__(
        self,
        input_channels=3,
        backbone="resnet18",
        pretrained=False,
        output_activation="sigmoid_range",
        min_output=1.0e-4,
        output_min=0.5,
        output_max=6.0,
        backbone_padding_mode="zeros",
    ):
        super().__init__()
        if input_channels not in (1, 3):
            raise ValueError("input_channels must be 1 or 3.")
        if output_activation not in {"softplus", "identity", "sigmoid_range"}:
            raise ValueError("output_activation must be softplus, identity, or sigmoid_range.")
        self.output_activation = output_activation
        self.min_output = float(min_output)
        self.output_min = float(output_min)
        self.output_max = float(output_max)
        self.backbone_padding_mode = str(backbone_padding_mode).lower()
        if self.backbone_padding_mode not in {"zeros", "reflect", "replicate"}:
            raise ValueError("backbone_padding_mode must be zeros, reflect, or replicate.")
        if output_activation == "sigmoid_range" and self.output_min >= self.output_max:
            raise ValueError("output_min must be smaller than output_max.")

        backbone_ctor = getattr(models, backbone)
        try:
            weights = "DEFAULT" if pretrained else None
            base = backbone_ctor(weights=weights)
        except TypeError:
            base = backbone_ctor(pretrained=pretrained)

        if input_channels != 3:
            old = base.conv1
            base.conv1 = nn.Conv2d(input_channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
            if pretrained and input_channels == 1:
                with torch.no_grad():
                    base.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))

        in_features = base.fc.in_features
        if self.backbone_padding_mode != "zeros":
            for module in base.modules():
                if isinstance(module, nn.Conv2d) and any(module.padding):
                    module.padding_mode = self.backbone_padding_mode
            base.maxpool = SpatialPadMaxPool2d(base.maxpool, self.backbone_padding_mode)
        base.fc = nn.Linear(in_features, 1)
        self.base = base

    def _positive_output(self, raw):
        if self.output_activation == "sigmoid_range":
            return self.output_min + (self.output_max - self.output_min) * torch.sigmoid(raw)
        if self.output_activation == "softplus":
            return F.softplus(raw) + self.min_output
        return raw.clamp_min(self.min_output)

    def forward(self, x):
        raw = self.base(x).squeeze(1)
        pred_gsd_ft = self._positive_output(raw)
        mu_map = pred_gsd_ft.view(-1, 1, 1, 1)
        logvar_map = torch.zeros_like(mu_map)
        weight_map = torch.ones_like(mu_map)
        return {
            "features": None,
            "mu_map": mu_map,
            "logvar_map": logvar_map,
            "precision_map": weight_map,
            "weight_map": weight_map,
            "pred_gsd_ft": pred_gsd_ft,
        }
