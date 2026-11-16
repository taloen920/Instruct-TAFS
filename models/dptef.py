"""Dual-Path Text Embedding Fusion (DPTEF), paper Equations (1)-(5)."""

import torch
from torch import nn


class DPTEF(nn.Module):
    """Pool, project, and fuse instruction and description representations."""

    def __init__(self, input_dim, text_dim=256, hidden_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.text_norm = nn.LayerNorm(input_dim)
        self.text_projection = nn.Linear(input_dim, text_dim)
        self.proj_instr = nn.Linear(text_dim, hidden_dim)
        self.proj_desc = nn.Linear(text_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        self.skip_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.output_proj = nn.Sequential(nn.Linear(hidden_dim, text_dim), nn.LayerNorm(text_dim))
        nn.init.xavier_uniform_(self.text_projection.weight)
        nn.init.zeros_(self.text_projection.bias)

    def pool_instruction(self, hidden_states, valid_tokens):
        """Mean-pool only valid text tokens, excluding padding and special markers."""
        projected = self.text_projection(self.text_norm(hidden_states))
        weights = valid_tokens.to(projected.dtype).unsqueeze(-1)
        return (projected * weights).sum(1) / weights.sum(1).clamp_min(1)

    def pool_description(self, hidden_states, eos_positions):
        """Project the final EOS state, or the last valid state when EOS is absent."""
        batch = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return self.text_projection(self.text_norm(hidden_states[batch, eos_positions]))

    def fuse_embeddings(self, instruction_embedding, description_embedding):
        """Apply bidirectional interaction, an adaptive gate, and output projection."""
        instruction = self.proj_instr(instruction_embedding)
        description = self.proj_desc(description_embedding)
        joined = torch.cat((instruction, description), dim=-1)

        # The singleton axis is the token axis; samples must never attend to each other.
        instruction_tokens = instruction.unsqueeze(1)
        description_tokens = description.unsqueeze(1)
        read_description, _ = self.cross_attn(
            instruction_tokens, description_tokens, description_tokens, need_weights=False
        )
        read_instruction, _ = self.cross_attn(
            description_tokens, instruction_tokens, instruction_tokens, need_weights=False
        )
        interaction = 0.5 * (
            self.cross_attn_norm(instruction + read_description.squeeze(1))
            + self.cross_attn_norm(description + read_instruction.squeeze(1))
        )
        gate = self.gate(joined)
        fused = gate * interaction + (1 - gate) * self.skip_proj(joined)
        return self.output_proj(fused)

    def forward(self, instruction_states, instruction_mask, description_states, eos_positions):
        instruction = self.pool_instruction(instruction_states, instruction_mask)
        description = self.pool_description(description_states, eos_positions)
        return self.fuse_embeddings(instruction, description)
