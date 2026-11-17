import numpy as np
import torch
import torch.nn.functional as F

def local_class_target(target, level_size, model):
    """Paper's local classification: equal intervals on the complete log range."""
    lower = model.base_gsd * model.mantissa_min
    upper = model.base_gsd * model.mantissa_max * (2 ** (model.num_classes - 1))
    position = torch.log(target.clamp(lower, upper) / lower) / np.log(upper / lower)
    return torch.floor(position * level_size).long().clamp(0, level_size - 1)

def select_leaf_vectors(leaf_vectors, exponent):
    # B,C,V,H,W -> B,V,H,W, selecting a per-example exponent.
    batch, _, channels, height, width = leaf_vectors.shape
    index = exponent.view(batch, 1, 1, 1, 1).expand(-1, 1, channels, height, width)
    return torch.gather(leaf_vectors, 1, index).squeeze(1)

def tree_losses(outputs, target, model, loss_cfg):
    """Reference Le + Lm objective, with the paper's 10% adjacent-path augmentation."""
    class_losses = []
    for logits in outputs["level_logits"][1:]:  # root has only one possible class
        level_size = logits.shape[1]
        cls = local_class_target(target, level_size, model)
        # Each of the M^2 (8x8 for a 256px input) outputs receives the local
        # exponent supervision; the paper averages the M^2 GSD estimates only
        # after the tree and mantissa computations.
        class_losses.append(
            F.cross_entropy(logits, cls[:, None, None].expand(-1, logits.shape[2], logits.shape[3]))
        )
    class_loss = torch.stack(class_losses).mean()

    true_exponent = model.exponent_for_target(target)
    true_vector = F.normalize(select_leaf_vectors(outputs["leaf_vectors"], true_exponent), p=2, dim=1)
    true_mantissa = model.mantissa_head(true_vector)
    target_mantissa = model.mantissa_for_exponent(target, true_exponent).view(-1, 1, 1, 1)
    mantissa_loss = F.l1_loss(true_mantissa, target_mantissa.expand_as(true_mantissa))

    # The paper adds, for 10% of training samples, the two possible ±1
    # class-error paths and their adjusted mantissas.  Do not apply an Lm
    # target to invalid leaves beyond either end of the tree.
    neighbor_fraction = float(loss_cfg.get("neighbor_error_fraction", 0.10))
    neighbor_loss = mantissa_loss.detach() * 0.0
    if neighbor_fraction > 0:
        augmented = torch.rand_like(target) < neighbor_fraction
        predictions, expected_values = [], []
        for direction in (-1, 1):
            neighbor_exp = true_exponent + direction
            usable = augmented & (neighbor_exp >= 0) & (neighbor_exp < model.num_classes)
            if not torch.any(usable):
                continue
            # The adjusted neighbouring-path mantissa is intentionally not
            # clipped to the canonical [0.75, 1.5] target interval.  E.g., a
            # class-k image assigned to k-1 needs twice its canonical
            # mantissa; this is the paper's one-to-many correction.
            vectors = F.normalize(select_leaf_vectors(outputs["leaf_vectors"], neighbor_exp.clamp(0, model.num_classes - 1)), p=2, dim=1)
            predicted = model.mantissa_head(vectors)
            expected = model.mantissa_for_exponent(target, neighbor_exp.clamp(0, model.num_classes - 1))
            predictions.append(predicted[usable])
            expected_values.append(expected[usable].view(-1, 1, 1, 1).expand_as(predicted[usable]))
        if predictions:
            neighbor_loss = F.l1_loss(torch.cat(predictions), torch.cat(expected_values))

    direct_weight = float(loss_cfg.get("direct_log_loss_weight", 0.0))
    direct_loss = F.smooth_l1_loss(torch.log(outputs["pred_gsd"].clamp_min(1e-8)), torch.log(target.clamp_min(1e-8)), beta=0.1)
    total = (
        float(loss_cfg.get("classification_weight", 1.0)) * class_loss
        + float(loss_cfg.get("mantissa_weight", 10.0)) * mantissa_loss
        + float(loss_cfg.get("neighbor_mantissa_weight", 1.0)) * neighbor_loss
        + direct_weight * direct_loss
    )
    return total, {
        "class_loss": class_loss.detach(), "mantissa_loss": mantissa_loss.detach(),
        "neighbor_loss": neighbor_loss.detach(), "direct_loss": direct_loss.detach(),
    }
