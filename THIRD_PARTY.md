# Dependencies and attribution

RT-CNN is a PyTorch reimplementation of Lee and Sull (2019), Regression Tree CNN for Estimation of Ground Sampling Distance Based on Floating-Point Representation. These are the weights trained for the reported comparisons, not weights supplied by the reference authors. The binomial tree, norm-based class routing, unrestricted mantissa readout, and adjacent-path loss are implemented in `rtcnn.py` and `tree_loss.py`.

The ResNet-18 architecture comes from torchvision. UREP uses its ResNet-18 components with an FPN, GroupNorm, patch heads, and learned or uniform aggregation. PyTorch and torchvision retain their upstream licenses; dependency terms are not replaced by this package. Dataset imagery has separate upstream rights described in the ModelScope data cards.

Project-authored code and trained weights are covered by the UREP Academic Non-Commercial License in LICENSE. It grants rights only in the covered project materials and does not override third-party software, source imagery, source metadata or applicable pretrained-model terms.
