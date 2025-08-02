import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    A features extractor that uses a Transformer encoder to process the sequence of market data.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, n_head: int = 4, n_layers: int = 2):
        super().__init__(observation_space, features_dim)

        n_input_features = observation_space.shape[1]

        encoder_layer = nn.TransformerEncoderLayer(d_model=n_input_features, nhead=n_head)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # The output of the transformer will be (seq_len, batch_size, n_input_features)
        # We need to flatten it to (batch_size, seq_len * n_input_features)
        # and then pass it through a linear layer to get the desired features_dim

        # To calculate the flattened size, we need the sequence length, which is part of the observation shape
        seq_len = observation_space.shape[0]
        flattened_size = seq_len * n_input_features

        self.linear = nn.Sequential(
            nn.Linear(flattened_size, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # The input to the transformer encoder should be (seq_len, batch_size, n_input_features)
        # The observations from the env are (batch_size, seq_len, n_input_features)
        # So we need to permute the dimensions
        observations = observations.permute(1, 0, 2)

        encoded_features = self.transformer_encoder(observations)

        # Permute back and flatten
        encoded_features = encoded_features.permute(1, 0, 2)
        flattened_features = th.flatten(encoded_features, start_dim=1)

        return self.linear(flattened_features)

class TransformerPolicy(ActorCriticPolicy):
    """
    A policy that uses the TransformerFeaturesExtractor.
    """
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super(TransformerPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=TransformerFeaturesExtractor,
            *args,
            **kwargs,
        )
