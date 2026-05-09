import numpy as np
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, maxlen: int = 750, scale_factor: float = 1.0):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size * scale_factor)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class DualScaleTemporalEncoder(nn.Module):
    """
    Unified temporal encoder that captures both fine-grained local motion 
    and long-range temporal dependencies without relying on sequence-length branching.
    """
    def __init__(self, embedding_dim, num_heads, dropout):
        super(DualScaleTemporalEncoder, self).__init__()
        
        # Local scale: Depthwise 1D Convolutions for short-term temporal dynamics
        self.local_encoder = nn.Conv1d(
            in_channels=embedding_dim, 
            out_channels=embedding_dim, 
            kernel_size=5, 
            padding=2, 
            groups=embedding_dim # Depthwise convolution for efficiency
        )
        self.local_norm = nn.LayerNorm(embedding_dim)
        
        # Global scale: Transformer Self-Attention for long-range dependencies
        self.global_encoder = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dropout=dropout, 
            activation='gelu',
            batch_first=False
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        x shape: [seq_len, batch, dim]
        """
        seq_len, batch, dim = x.shape
        
        # Local processing
        local_x = x.permute(1, 2, 0)  # [batch, dim, seq_len]
        local_x = self.local_encoder(local_x)
        local_x = local_x.permute(2, 0, 1)  # [seq_len, batch, dim]
        local_x = self.local_norm(local_x + x)  # Residual connection
        
        # Global processing
        global_x = self.global_encoder(local_x)
        
        # Fusion
        fused = self.fusion(torch.cat([local_x, global_x], dim=-1))
        
        return fused

class GaussianHistoryBank(nn.Module):
    """
    Hybrid Gaussian Latent History.

    A global learnable action-prototype dictionary (mu_prior, log_var_prior, w_prior)
    is adapted per input segment via data-conditioned deltas:

        mu_k(F_t)      = mu_prior_k       + g * Delta_mu_k(ctx(F_t))
        log_var_k(F_t) = log_var_prior_k  + g * Delta_logvar_k(ctx(F_t))
        w_k(F_t)       = w_prior_k        + g * Delta_w_k(ctx(F_t))

    where ctx(F_t) is a mean+max pooled summary of the current segment and g is a
    learnable scalar gate (sigmoid). Sampling uses the reparameterization trick.
    """
    def __init__(self, num_gaussians, embedding_dim, init_log_var=-2.0):
        super(GaussianHistoryBank, self).__init__()
        self.K = num_gaussians
        self.D = embedding_dim

        # Global prior: action prototype dictionary
        self.mu_prior = nn.Parameter(torch.randn(num_gaussians, embedding_dim) * 0.02)
        self.log_var_prior = nn.Parameter(torch.full((num_gaussians, embedding_dim), init_log_var))
        self.weight_logits_prior = nn.Parameter(torch.zeros(num_gaussians))

        # Context summarization head (mean+max pool -> compact context vector)
        self.context_proj = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim)
        )

        # Per-input adaptation heads (zero-init so model starts at the prior)
        self.delta_mu_head = nn.Linear(embedding_dim, num_gaussians * embedding_dim)
        self.delta_logvar_head = nn.Linear(embedding_dim, num_gaussians * embedding_dim)
        self.delta_weight_head = nn.Linear(embedding_dim, num_gaussians)
        for layer in (self.delta_mu_head, self.delta_logvar_head, self.delta_weight_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Adaptation gate (sigmoid(-2) ~ 0.12, so prior dominates early in training)
        self.adapt_gate = nn.Parameter(torch.tensor(-2.0))

    def summarize(self, x):
        # x: [seq_len, batch, dim] -> [batch, dim]
        mean_pool = x.mean(dim=0)
        max_pool = x.max(dim=0)[0]
        return self.context_proj(torch.cat([mean_pool, max_pool], dim=-1))

    def sample(self, x):
        seq_len, batch, dim = x.shape

        ctx = self.summarize(x)                                              # [B, D]
        delta_mu = self.delta_mu_head(ctx).view(batch, self.K, self.D)       # [B, K, D]
        delta_logvar = self.delta_logvar_head(ctx).view(batch, self.K, self.D)
        delta_weight = self.delta_weight_head(ctx)                           # [B, K]

        gate = torch.sigmoid(self.adapt_gate)
        mu = self.mu_prior.unsqueeze(0) + gate * delta_mu                    # [B, K, D]
        log_var = self.log_var_prior.unsqueeze(0) + gate * delta_logvar
        log_var = log_var.clamp(min=-10.0, max=2.0)                          # numerical stability
        weight_logits = self.weight_logits_prior.unsqueeze(0) + gate * delta_weight

        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std                                               # [B, K, D]
        else:
            z = mu                                                           # deterministic at inference
        weights = F.softmax(weight_logits, dim=-1)                           # [B, K]
        return z, weights

    def kl_divergence(self):
        # KL( N(mu_prior, sigma_prior^2) || N(0, I) ) averaged over K and D so that
        # kl_weight stays meaningful when K or D changes.
        return -0.5 * torch.mean(
            1 + self.log_var_prior - self.mu_prior.pow(2) - self.log_var_prior.exp()
        )


class GaussianCrossAttention(nn.Module):
    """
    Cross-attends current encoded features against the (input-conditioned) Gaussian
    Latent History bank. Each batch element gets its own adapted set of K primitives.
    Output is a residual-augmented version of the input features.
    """
    def __init__(self, embedding_dim, num_heads, num_gaussians, dropout=0.1):
        super(GaussianCrossAttention, self).__init__()
        self.bank = GaussianHistoryBank(num_gaussians, embedding_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        # Learnable residual gate: starts at 0 so the module is a no-op at init
        # (model behaves identically to baseline; can only learn to add GLH if useful).
        self.residual_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: [seq_len, batch, dim]
        z, weights = self.bank.sample(x)                              # z: [B, K, D], w: [B, K]
        memory = (z * weights.unsqueeze(-1)).permute(1, 0, 2)         # [K, B, D]

        attn_out, _ = self.cross_attn(query=x, key=memory, value=memory)
        x = self.norm(x + self.residual_gate * self.dropout(attn_out))
        return x

    def kl_loss(self):
        return self.bank.kl_divergence()


class GTANRefinement(nn.Module):
    """
    Faithful GTAN (Long et al., CVPR 2019) adaptation for HAT.

    Implements paper Eq. (1) and Eq. (4):
      - Per-anchor sigma_a is PREDICTED from encoder features via a 1D conv head
        (paper Sec 3.2: "sigma_t^j is learnt via a 1D convolutional layer on a
        3 x D^j feature map cell"), then squashed to (0, 1) via sigmoid.
      - Gaussian pooling uses paper Eq. (1) weights in normalized [0, 1]
        coordinates (p_i = i/T, mu_t = t/T) and aggregates encoded features
        per Eq. (4).
      - Centers default to the segment midpoint (paper Eq. 2: a_c = (t+0.5)/T;
        HAT anchors are scale-only, so all anchors share the midpoint reference,
        with a small learnable offset for refinement).

    Adaptation notes (where we diverge from the paper):
      - HAT operates on a single segment with K scale-only anchors rather than
        per-position cells across 8 anchor layers, so the sigma-predictor pools
        across time after the conv instead of producing one sigma per cell.
      - Gaussian pooling is fused as a residual to HAT's existing transformer
        decoder output rather than replacing it; this keeps HAT's content-based
        attention and adds the geometry-aware GTAN feature stream alongside.
      - The Gaussian Kernel Grouping algorithm (paper Sec 3.3) and the overlap
        parameter loss (paper Eq. 9) are not included here; they are separate
        additions to consider if the basic GTAN integration moves the needle.
    """
    def __init__(self, embedding_dim, anchor_scales, segment_size, kernel_size=3):
        super(GTANRefinement, self).__init__()
        self.num_anchors = len(anchor_scales)
        self.embedding_dim = embedding_dim
        self.segment_size = segment_size

        # Paper Sec 3.2: 1D conv head predicting sigma per anchor.
        self.sigma_conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=self.num_anchors,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        # Initialize the conv bias so each anchor's predicted sigma starts near
        # its natural scale ((scale/2)/T in normalized units).
        with torch.no_grad():
            target_sigma = torch.tensor(
                [(float(s) / 2.0) / segment_size for s in anchor_scales],
                dtype=torch.float32
            ).clamp(min=1e-3, max=0.99)
            bias_init = torch.log(target_sigma / (1.0 - target_sigma))   # logit
            self.sigma_conv.bias.copy_(bias_init)
            # Zero the conv weights so initial sigma is determined by bias only;
            # the conv learns to perturb sigma based on local content.
            self.sigma_conv.weight.zero_()

        # Centers in normalized [0, 1]; learnable offset bounded to (-0.1, +0.1)
        # via tanh so centers stay near segment midpoint.
        self.center_offset_logits = nn.Parameter(torch.zeros(self.num_anchors))

    def predict_sigma(self, encoded_x):
        """encoded_x: [T, B, D] -> sigma: [B, K] in (0, 1) normalized units."""
        x = encoded_x.permute(1, 2, 0)                    # [B, D, T]
        feats = self.sigma_conv(x)                        # [B, K, T]
        feats = feats.mean(dim=-1)                        # [B, K] (global temporal pool)
        return torch.sigmoid(feats)

    def get_centers(self):
        # Centers in normalized [0, 1], near 0.5
        return 0.5 + 0.1 * torch.tanh(self.center_offset_logits)

    def gaussian_pool(self, encoded_x):
        """encoded_x: [T, B, D] -> pooled: [K, B, D] (paper Eq. 1 + Eq. 4)."""
        T, B, D = encoded_x.shape
        device, dtype = encoded_x.device, encoded_x.dtype

        sigma = self.predict_sigma(encoded_x).clamp(min=1e-2)      # [B, K]
        centers = self.get_centers().to(device=device, dtype=dtype)  # [K]

        # Position grid in normalized [0, 1] (paper: p_i = i/T)
        denom = max(T - 1, 1)
        p = torch.arange(T, device=device, dtype=dtype) / denom    # [T]

        diff = p.view(1, 1, -1) - centers.view(1, -1, 1)           # [1, K, T]
        log_weights = -0.5 * (diff ** 2) / (sigma.unsqueeze(-1) ** 2)  # [B, K, T]
        weights = F.softmax(log_weights, dim=-1)                   # [B, K, T]

        x = encoded_x.permute(1, 0, 2)                             # [B, T, D]
        pooled = torch.bmm(weights, x)                             # [B, K, D]
        return pooled.permute(1, 0, 2)                             # [K, B, D]

    @torch.no_grad()
    def get_anchor_params(self, encoded_x=None, seq_len=None):
        """For logging. Returns (centers_frames, sigmas_frames). If encoded_x is
        given, sigmas are the input-conditioned predictions averaged over batch;
        otherwise the bias-init estimate is used."""
        if seq_len is None:
            seq_len = self.segment_size
        centers = (self.get_centers() * (seq_len - 1)).detach().cpu()
        if encoded_x is not None:
            sigma = self.predict_sigma(encoded_x).mean(dim=0)
        else:
            sigma = torch.sigmoid(self.sigma_conv.bias)
        sigmas_frames = (sigma * seq_len).detach().cpu()
        return centers, sigmas_frames


class MYNET(torch.nn.Module):
    def __init__(self, opt):
        super(MYNET, self).__init__()
        self.n_feature = opt["feat_dim"] 
        n_class = opt["num_of_class"]
        n_embedding_dim = opt["hidden_dim"]
        n_enc_layer = opt["enc_layer"]
        n_enc_head = opt["enc_head"]
        n_dec_layer = opt["dec_layer"]
        n_dec_head = opt["dec_head"]
        n_seglen = opt["segment_size"]
        self.anchors = opt["anchors"]
        self.anchors_stride = []
        dropout = 0.3
        self.best_loss = 1000000
        self.best_map = 0
        
        # Enhanced feature reduction with dynamic dropout
        self.feature_reduction_rgb = nn.Sequential(
            nn.Linear(self.n_feature//2, n_embedding_dim//2),
            nn.GELU(),
            nn.LayerNorm(n_embedding_dim//2),
            nn.Dropout(dropout * 0.5)
        )
        self.feature_reduction_flow = nn.Sequential(
            nn.Linear(self.n_feature//2, n_embedding_dim//2),
            nn.GELU(),
            nn.LayerNorm(n_embedding_dim//2),
            nn.Dropout(dropout * 0.5)
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            n_embedding_dim, 
            dropout=dropout * 0.5,
            maxlen=400,
            scale_factor=0.5
        )
        
        # Unified Dual-Scale Temporal Encoder
        self.temporal_encoder = DualScaleTemporalEncoder(
            embedding_dim=n_embedding_dim,
            num_heads=n_enc_head,
            dropout=dropout
        )
        
        # Main encoder (adaptive layers)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embedding_dim, 
                nhead=n_enc_head, 
                dropout=dropout * (0.5 if i < 2 else 1.0),  # Lower dropout for initial layers
                activation='gelu'
            ) for i in range(n_enc_layer)
        ])
        self.encoder_norm = nn.LayerNorm(n_embedding_dim)
        
        # GTAN (Long et al., CVPR 2019) faithful adaptation: input-conditioned
        # per-anchor sigma predicted via a 1D conv head, used for Gaussian
        # pooling. Produces a parallel geometry-aware feature stream fused with
        # the decoder output as a residual, with per-anchor learnable scale.
        self.gtan_refinement = GTANRefinement(
            embedding_dim=n_embedding_dim,
            anchor_scales=self.anchors,
            segment_size=n_seglen
        )
        self.gtan_fusion_scale = nn.Parameter(torch.full((len(self.anchors),), 0.1))

        # Decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=n_embedding_dim, 
                nhead=n_dec_head, 
                dropout=dropout, 
                activation='gelu'
            ), 
            n_dec_layer, 
            nn.LayerNorm(n_embedding_dim)
        )
        

        
        # Enhanced classification and regression heads
        self.classifier = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim), 
            nn.GELU(), 
            nn.LayerNorm(n_embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(n_embedding_dim, n_class)
        )
        self.regressor = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim), 
            nn.GELU(), 
            nn.LayerNorm(n_embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(n_embedding_dim, 2)
        )
        
        self.decoder_token = nn.Parameter(torch.Tensor(len(self.anchors), 1, n_embedding_dim))
        nn.init.normal_(self.decoder_token, std=0.01)
        
        # Additional normalization layers
        self.norm1 = nn.LayerNorm(n_embedding_dim)
        self.norm2 = nn.LayerNorm(n_embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.relu = nn.ReLU(True)
        self.softmaxd1 = nn.Softmax(dim=-1)

    def forward(self, inputs):
        # Enhanced feature processing
        inputs = inputs.float()
        base_x_rgb = self.feature_reduction_rgb(inputs[:,:,:self.n_feature//2])
        base_x_flow = self.feature_reduction_flow(inputs[:,:,self.n_feature//2:])
        base_x = torch.cat([base_x_rgb, base_x_flow], dim=-1)
        
        base_x = base_x.permute([1,0,2])  # seq_len x batch x featsize
        seq_len = base_x.shape[0]
        
        # Apply positional encoding
        pe_x = self.positional_encoding(base_x)
        
        # Unified Dual-Scale Temporal Processing
        temporal_features = self.temporal_encoder(pe_x)
        
        # Standard encoder processing
        encoded_x = temporal_features
        for layer in self.encoder_layers:
            encoded_x = layer(encoded_x)
        
        # Apply encoder normalization
        encoded_x = self.encoder_norm(encoded_x)
        encoded_x = self.norm1(encoded_x)

        # Standard decoder cross-attention (unrestricted, baseline behavior).
        decoder_token = self.decoder_token.expand(-1, encoded_x.shape[1], -1)
        decoded_x = self.decoder(decoder_token, encoded_x)

        # GTAN-style parallel: Gaussian-pooled feature per anchor, fused as a
        # learnable residual. `gauss_pooled[a, b, :]` is the Gaussian-weighted
        # average of encoded_x along the time axis for anchor a.
        gauss_pooled = self.gtan_refinement.gaussian_pool(encoded_x)        # [K, B, D]
        decoded_x = decoded_x + self.gtan_fusion_scale.view(-1, 1, 1) * gauss_pooled
        
        # Add residual connection and normalization
        decoded_x = self.norm2(decoded_x + self.dropout1(decoder_token))
        
        decoded_x = decoded_x.permute([1, 0, 2])
        
        anc_cls = self.classifier(decoded_x)
        anc_reg = self.regressor(decoded_x)
        
        return anc_cls, anc_reg


class SuppressNet(torch.nn.Module):
    def __init__(self, opt):
        super(SuppressNet, self).__init__()
        n_class=opt["num_of_class"]-1
        n_seglen=opt["segment_size"]
        n_embedding_dim=2*n_seglen
        dropout=0.3
        self.best_loss=1000000
        self.best_map=0
        # FC layers for the 2 streams
        
        self.mlp1 = nn.Linear(n_seglen, n_embedding_dim)
        self.mlp2 = nn.Linear(n_embedding_dim, 1)
        self.norm = nn.InstanceNorm1d(n_class)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        #inputs - batch x seq_len x class
        
        base_x = inputs.permute([0,2,1])
        base_x = self.norm(base_x)
        x = self.relu(self.mlp1(base_x))
        x = self.sigmoid(self.mlp2(x))
        x = x.squeeze(-1)
        
        return x
