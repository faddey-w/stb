import torch.nn
from strateobots.ai import coding
from strateobots.ai.nn.embedder import Embedder


class Set2SetPredictor(torch.nn.Module):
    def __init__(self, object_dim):
        super(Set2SetPredictor, self).__init__()

        self.bot_ctl_embedder = Embedder(coding.bot_full_coder.dim + coding.control_coder.dim, object_dim)
        self.bullet_embedder = Embedder(coding.bullet_coder.dim, object_dim)
        self.ray_embedder = Embedder(coding.bullet_coder.dim, object_dim)

        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                object_dim,
                nhead=4,
                dim_feedforward=4 * object_dim,
                dropout=0.1,
                activation="relu",
            ),
            num_layers=3,
            norm=torch.nn.LayerNorm(object_dim),
        )
        self.bot_predictor = torch.nn.Sequential(
            torch.nn.Linear(object_dim, 2 * object_dim, bias=True),
            torch.nn.LeakyReLU(),
        )
        self.double()
        self._reset_parameters()

    def forward(self, state_batch: coding.WorldStateCodes, mask_batch: coding.WorldStateCodes):
        return self.compute_internal_repr(state_batch, mask_batch)

    def compute_internal_repr(
        self, state_batch: coding.WorldStateCodes, mask_batch: coding.WorldStateCodes
    ):
        bot_ctl_embeddings = self.bot_ctl_embedder(torch.cat(
            [state_batch.bots,state_batch.controls], dim=2
        ))
        bullet_embeddings = self.bullet_embedder(state_batch.bullets)
        ray_embeddings = self.ray_embedder(state_batch.rays)

        all_embeddings = torch.cat([bot_ctl_embeddings, bullet_embeddings, ray_embeddings], dim=0)
        # bots mask should be the same as controls mask
        all_mask = torch.cat([mask_batch.bots, mask_batch.bullets, mask_batch.rays], dim=1)

        memory = self.encoder(all_embeddings, src_key_padding_mask=all_mask)
        return memory

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
