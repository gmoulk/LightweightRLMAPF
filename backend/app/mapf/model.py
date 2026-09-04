import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pogema import pogema_v0
from pogema import GridConfig

class CommTransformerNet(nn.Module):
    def __init__(self, obs_shape=(3, 11, 11), n_actions=5, seq_len=10, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim

        # 1. LOCAL FEATURE EXTRACTOR (CNN)
        # We use a slightly deeper CNN to capture better spatial features for larger maps
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2), # 11x11 -> 5x5
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Flatten()
        )

        # 2. PROJECTION LAYER
        # Fuses CNN features (1600) + Goal Vector (2) into Embedding Space
        self.projection = nn.Linear(64 * 5 * 5 + 2, embed_dim)

        # 3. MULTI-HEAD SELECTIVE COMMUNICATION
        # 8 heads allow an agent to attend to 8 different 'types' of neighbor threats
        self.comm_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=0.1,  # Helps prevent 'Communication Overfitting' at high agent counts
            batch_first=True
        )

        # 4. STABILIZATION LAYERS
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        # Feed-Forward Network (FFN) to process the social context
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # 5. OUTPUT HEADS
        self.policy_head = nn.Linear(embed_dim, n_actions)
        self.value_head = nn.Linear(embed_dim, 1)

        # 6. ORTHOGONAL INITIALIZATION
        self._apply_init()

    def _apply_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('leaky_relu'))
                nn.init.constant_(m.bias, 0)
        # Policy head gain should be very small to prioritize Expert IL at the start
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)

    def forward(self, obs_seq, goal_seq):
        """
        obs_seq:  (num_agents, seq_len, 3, 11, 11)
        goal_seq: (num_agents, seq_len, 2)
        """
        num_agents, seq_len, C, H, W = obs_seq.shape

        # A. Process current state (ignore past sequence for now to save CPU)
        curr_obs = obs_seq[:, -1]
        curr_goal = goal_seq[:, -1]

        cnn_feats = self.feature_extractor(curr_obs)

        # B. Fusion & Projection
        combined = torch.cat([cnn_feats, curr_goal], dim=-1)
        x = self.projection(combined) # (num_agents, embed_dim)

        # C. Multi-Agent Communication (Transformer Encoder Layer logic)
        # Prepare for MultiheadAttention: (Batch=1, Seq=num_agents, Dim=embed_dim)
        x_in = x.unsqueeze(0)

        # 1. Self-Attention Block
        attn_out, weights = self.comm_attn(x_in, x_in, x_in)
        x_norm1 = self.layernorm1(x_in + attn_out) # Residual + Norm

        # 2. Feed-Forward Block (Deepens the "social understanding")
        ffn_out = self.ffn(x_norm1)
        x_context = self.layernorm2(x_norm1 + ffn_out).squeeze(0) # Residual + Norm

        # D. Final Predictions
        logits = self.policy_head(x_context)
        values = self.value_head(x_context)

        return logits, values, weights