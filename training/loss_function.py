import torch
import torch.nn.functional as F


def compute_forge_loss(logits, targets, h_states, embedding_layer, vocab_size, alpha):
    """
    Compute the composite Forge loss with three components:
    
    1. Cross-Entropy Loss (standard language modeling)
    2. Hidden State Norm Regularization (stabilizes activation magnitude)
    3. Cosine Similarity Alignment (encourages representation alignment with target embeddings)
    
    Args:
        logits: Model output logits [B, T, vocab_size]
        targets: Target token IDs [B, T]
        h_states: Hidden states from final layer [B, T, d_model]
        embedding_layer: nn.Embedding layer for target tokens
        vocab_size: Size of vocabulary
        alpha: Dynamic scaling factor from ForgeController
    
    Returns:
        total_loss: Combined loss scalar
        loss_components: Dictionary of individual loss terms
    """
    # 1. Cross-Entropy Loss
    ce_loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1)
    )

    # 2. Hidden State Norm Regularization
    h_norm = torch.sqrt(torch.mean(h_states ** 2))

    # 3. Cosine Similarity Alignment
    with torch.no_grad():
        target_emb = embedding_layer(targets)  # [B, T, d_model]
    
    cos_sim = F.cosine_similarity(h_states, target_emb, dim=-1).mean()
    cos_loss = 1.0 - cos_sim

    # Combine with Forge scaling
    total_loss = (
        ce_loss
        + alpha * 0.15 * h_norm
        + alpha * 0.40 * cos_loss
    )

    loss_components = {
        "ce_loss": ce_loss.item(),
        "h_norm": h_norm.item(),
        "cos_sim": cos_sim.item(),
        "cos_loss": cos_loss.item(),
        "alpha": alpha
    }

    return total_loss, loss_components