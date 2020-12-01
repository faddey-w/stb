import torch.nn


class Embedder(torch.nn.Module):
    def __init__(self, object_vector_dim, embedding_dim):
        super(Embedder, self).__init__()
        self._embedding_dim = embedding_dim

        self._scaler = torch.nn.BatchNorm1d(object_vector_dim)
        self._fc = torch.nn.Linear(object_vector_dim, embedding_dim, bias=True)
        self._act = torch.nn.Sigmoid()

    def forward(self, object_vectors):
        shape = object_vectors.shape
        object_vectors_flat = object_vectors.view(-1, shape[-1])
        embedding_flat = self._act(self._fc(self._scaler(object_vectors_flat)))
        return embedding_flat.view(*shape[:-1], self._embedding_dim)

    def restore_scale(self, unscaled_object_vectors):
        ndim = len(unscaled_object_vectors.shape)
        expanded_shape = (1,) * (ndim-1) + (-1,)
        std = self._scaler.running_var.view(expanded_shape).detach()
        mean = self._scaler.running_mean.view(expanded_shape).detach()
        return unscaled_object_vectors * (std + self._scaler.eps) + mean
