"""Muon optimizer with QK-Clip and supporting attention utilities.

Implements the MuonClip optimizer, which applies Newton-Schulz orthogonalized
momentum updates to 2D+ parameters and standard momentum to 1D parameters, with
QK-Clip rescaling of attention query/key projections to bound attention logits.
Also provides standalone per-head QK-Clip helpers and an example multi-head
attention layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple, Callable


def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Orthogonalize a matrix via Newton-Schulz iteration.

    The input is normalized by its norm, transposed when it has more rows than
    columns so the iteration operates on the wider orientation, iterated with
    fixed quintic coefficients, and cast back to the original dtype.

    Args:
        G: Input matrix tensor.
        steps: Number of iteration steps.
        eps: Numerical-stability epsilon.

    Returns:
        Orthogonalized matrix with the same shape and dtype as G.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G.float()
    X /= (X.norm() + eps)

    if G.size(0) > G.size(1):
        X = X.T
        transposed = True
    else:
        transposed = False

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X.to(G.dtype)


class MuonClip(torch.optim.Optimizer):
    """
    MuonClip Optimizer - Combines Muon optimizer with QK-Clip for stable LLM training.
    
    This optimizer applies:
    1. Muon updates with Newton-Schulz orthogonalization for 2D+ parameters
    2. Standard momentum for 1D parameters
    3. QK-Clip to prevent attention logit explosion
    
    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 1e-3)
        momentum: Momentum coefficient μ (default: 0.95)
        weight_decay: Weight decay coefficient λ (default: 0.01)
        tau: QK-Clip threshold τ (default: 100.0)
        ns_steps: Newton-Schulz iteration steps (default: 5)
        eps: Numerical stability epsilon (default: 1e-7)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        tau: float = 100.0,
        ns_steps: int = 5,
        eps: float = 1e-7,
        update_clip: float = 5.0,
    ):
        """Initialize the optimizer and validate hyperparameters.

        Args:
            params: Parameters to optimize.
            lr: Learning rate.
            momentum: Momentum coefficient.
            weight_decay: Weight-decay coefficient.
            tau: QK-Clip logit threshold.
            ns_steps: Newton-Schulz iteration steps.
            eps: Numerical-stability epsilon.
            update_clip: Maximum ratio of the orthogonalized update norm to the
                gradient norm; 0 disables clipping.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 < tau:
            raise ValueError(f"Invalid tau value: {tau}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            tau=tau,
            ns_steps=ns_steps,
            eps=eps,
            update_clip=update_clip,
        )
        super(MuonClip, self).__init__(params, defaults)

        self.model = None
        self._mha_modules = []
        self._max_logits = {}
        self._hooks = []

    def set_model(self, model: nn.Module):
        """Register forward hooks on all MultiheadAttention layers for QK-Clip."""
        self.model = model
        self._mha_modules = []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.MultiheadAttention):
                self._mha_modules.append((name, mod))
                self._max_logits[id(mod)] = 0.0
                hook = mod.register_forward_hook(self._attn_logit_hook)
                self._hooks.append(hook)

    @staticmethod
    def _attn_logit_hook(module, input, output):
        """Forward hook estimating a module's max attention logit for QK-Clip.

        nn.MultiheadAttention returns (attn_output, attn_weights). When weights
        are available (post-softmax), the pre-softmax maximum is estimated as
        log(max_weight) + log(seq_len). When weights are None, the estimate
        falls back to the Frobenius norms of the Q/K projection blocks of
        in_proj_weight. The estimate is stored on module._muon_max_logit.
        """
        attn_out, attn_weights = output
        if attn_weights is not None:
            max_w = attn_weights.max().item()
            seq_len = attn_weights.shape[-1]
            if max_w > 0:
                module._muon_max_logit = math.log(max_w + 1e-10) + math.log(seq_len)
            else:
                module._muon_max_logit = 0.0
        else:
            if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                d = module.in_proj_weight.shape[1]
                q_weight = module.in_proj_weight[:d]
                k_weight = module.in_proj_weight[d:2*d]
                num_heads = module.num_heads
                d_k = d // num_heads
                q_norm = q_weight.norm().item()
                k_norm = k_weight.norm().item()
                module._muon_max_logit = q_norm * k_norm / (math.sqrt(d_k) * d)
            else:
                module._muon_max_logit = 0.0
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform a single optimization step, then apply QK-Clip.

        Parameters with 2 or more dimensions receive Newton-Schulz
        orthogonalized momentum updates with RMS scaling and optional update-norm
        clipping; 1D parameters receive standard momentum updates. After the
        parameter update, QK-Clip rescales attention Q/K projections.

        Args:
            closure: Optional closure that reevaluates the model and returns
                the loss.

        Returns:
            The loss value if a closure is provided, otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            ns_steps = group['ns_steps']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                param_state = self.state[p]

                if len(param_state) == 0:
                    param_state['momentum_buffer'] = torch.zeros_like(p.data)

                buf = param_state['momentum_buffer']

                buf.mul_(momentum).add_(grad)

                if p.ndim >= 2:
                    if p.ndim > 2:
                        original_shape = buf.shape
                        buf_2d = buf.view(buf.shape[0], -1)
                        orthogonal_update = newton_schulz(buf_2d, ns_steps, eps)
                        orthogonal_update = orthogonal_update.view(original_shape)
                    else:
                        orthogonal_update = newton_schulz(buf, ns_steps, eps)

                    n, m = p.shape[0], p.shape[1] if p.ndim > 1 else 1
                    rms_factor = math.sqrt(max(n, m) * 0.2)
                    orthogonal_update *= rms_factor

                    update_clip = group['update_clip']
                    if update_clip > 0:
                        grad_norm = grad.norm()
                        update_norm = orthogonal_update.norm()
                        max_norm = grad_norm * update_clip
                        if update_norm > max_norm and max_norm > 0:
                            orthogonal_update *= max_norm / update_norm

                    p.data.add_(orthogonal_update + weight_decay * p.data, alpha=-lr)

                else:
                    p.data.add_(buf + weight_decay * p.data, alpha=-lr)

        self._apply_qk_clip()

        return loss
    
    def _apply_qk_clip(self):
        """Apply QK-Clip to nn.MultiheadAttention layers.

        For nn.MultiheadAttention, in_proj_weight is [3*d, d] where
        rows [0:d] = Q weights, [d:2d] = K weights, [2d:3d] = V weights.
        When max attention logit exceeds tau, scale down Q and K weights
        by sqrt(tau/max_logit) to prevent logit explosion.
        """
        if not self._mha_modules:
            return

        tau = self.param_groups[0]['tau']

        for name, mha in self._mha_modules:
            max_logit = getattr(mha, '_muon_max_logit', 0.0)
            if max_logit <= tau:
                continue

            gamma = tau / max_logit
            sqrt_gamma = math.sqrt(gamma)

            if hasattr(mha, 'in_proj_weight') and mha.in_proj_weight is not None:
                d = mha.in_proj_weight.shape[1]
                mha.in_proj_weight.data[:d] *= sqrt_gamma
                mha.in_proj_weight.data[d:2*d] *= sqrt_gamma
                if mha.in_proj_bias is not None:
                    mha.in_proj_bias.data[:d] *= sqrt_gamma
                    mha.in_proj_bias.data[d:2*d] *= sqrt_gamma


def apply_qk_clip_per_head(
    query_weights: torch.Tensor,
    key_weights: torch.Tensor,
    max_logits_per_head: List[float],
    tau: float = 100.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply per-head QK-Clip following Algorithm 1, lines 11-16.
    
    Args:
        query_weights: [d_model, d_model] Query projection weights
        key_weights: [d_model, d_model] Key projection weights  
        max_logits_per_head: List of max logits per head
        tau: Threshold for clipping
        
    Returns:
        Clipped query and key weights
    """
    d_model = query_weights.shape[0]
    num_heads = len(max_logits_per_head)
    d_k = d_model // num_heads
    
    clipped_query = query_weights.clone()
    clipped_key = key_weights.clone()
    
    for head_idx, max_logit in enumerate(max_logits_per_head):
        if max_logit > tau:
            gamma = tau / max_logit
            sqrt_gamma = math.sqrt(gamma)
            
            # Scale the query and key parameters for this attention head.
            start_idx = head_idx * d_k
            end_idx = start_idx + d_k
            
            clipped_query[:, start_idx:end_idx] *= sqrt_gamma
            clipped_key[:, start_idx:end_idx] *= sqrt_gamma
    
    return clipped_query, clipped_key


def apply_qk_clip_to_model(
    model: nn.Module,
    attention_logits: dict,
    tau: float = 100.0
) -> None:
    """
    Apply per-head QK-Clip to model parameters.
    
    Args:
        model: PyTorch model with attention layers
        attention_logits: Dict mapping layer names to per-head max logits
        tau: Clipping threshold
    """
    if hasattr(model, 'get_attention_layers'):
        attention_layers = model.get_attention_layers()
        
        for layer_name, attention_layer in attention_layers:
            if layer_name in attention_logits:
                max_logits_per_head = attention_logits[layer_name]
                
                if hasattr(attention_layer, 'query') and hasattr(attention_layer, 'key'):
                    query_weight = attention_layer.query.weight.data
                    key_weight = attention_layer.key.weight.data
                    
                    clipped_q, clipped_k = apply_qk_clip_per_head(
                        query_weight, key_weight, max_logits_per_head, tau
                    )
                    
                    attention_layer.query.weight.data = clipped_q
                    attention_layer.key.weight.data = clipped_k


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer with QK-Clip support.
    
    This implementation tracks maximum attention logits for QK-Clip.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.max_logits = 0.0
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Project inputs into per-head query, key, and value tensors.
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Record the maximum logit for QK-Clip updates.
        with torch.no_grad():
            self.max_logits = torch.max(scores).item()
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.output(context)
        
        return output


def create_optimizer_separate_groups(model: nn.Module, lr: float = 1e-3, **kwargs) -> Tuple[MuonClip, torch.optim.AdamW]:
    """
    Create MuonClip for 2D+ parameters and AdamW for 1D parameters.
    
    This is the recommended approach from the paper for optimal performance.
    
    Args:
        model: PyTorch model
        lr: Learning rate
        **kwargs: Additional arguments for MuonClip
        
    Returns:
        Tuple of (MuonClip optimizer, AdamW optimizer)
    """
    muon_params = [p for p in model.parameters() if p.ndim >= 2]
    adamw_params = [p for p in model.parameters() if p.ndim < 2]
    
    muon_optimizer = MuonClip(muon_params, lr=lr, **kwargs)
    adamw_optimizer = torch.optim.AdamW(adamw_params, lr=lr * 0.3, weight_decay=kwargs.get('weight_decay', 0.01))
    
    muon_optimizer.set_model(model)
    
    return muon_optimizer, adamw_optimizer


def train_step(
    model: nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    muon_opt: MuonClip,
    adamw_opt: Optional[torch.optim.AdamW] = None,
    criterion: Optional[nn.Module] = None
) -> float:
    """
    Single training step with MuonClip.
    
    Args:
        model: PyTorch model
        data: Input data
        target: Target labels
        muon_opt: MuonClip optimizer
        adamw_opt: Optional AdamW optimizer for 1D params
        criterion: Loss function (default: CrossEntropyLoss)
        
    Returns:
        Loss value
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    output = model(data)
    loss = criterion(output, target)
    
    muon_opt.zero_grad()
    if adamw_opt:
        adamw_opt.zero_grad()
    
    loss.backward()
    
    muon_opt.step()
    if adamw_opt:
        adamw_opt.step()
    
    return loss.item()
