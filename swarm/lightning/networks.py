"""
Neural Network Architectures for Swarm RL

Provides modular network components that can be combined
or replaced with custom architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math


class MLPNetwork(nn.Module):
    """
    Multi-Layer Perceptron for flat observations.
    
    Args:
        input_dim: Input dimension (flattened observation)
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension (action dim or value)
        activation: Activation function
        output_activation: Final activation (None for linear)
    
    Example:
        >>> net = MLPNetwork(1024, [256, 256], 17)
        >>> x = torch.randn(32, 1024)
        >>> out = net(x)  # (32, 17)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = 'relu',
        output_activation: Optional[str] = None,
    ):
        super().__init__()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        if output_activation:
            layers.append(self._get_activation(output_activation))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(),
        }
        return activations.get(name, nn.ReLU())
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNNetwork(nn.Module):
    """
    Convolutional Neural Network for 2D density grid observations.
    
    Designed for the 32x32 density grid observations from SwarmEnv.
    
    Args:
        input_channels: Number of input channels (1 for density grid)
        output_dim: Output dimension
        conv_channels: List of conv layer channels
        kernel_sizes: List of kernel sizes
        hidden_dim: Hidden dimension after conv
    
    Example:
        >>> net = CNNNetwork(1, 17, [32, 64, 64], [3, 3, 3], 256)
        >>> x = torch.randn(32, 1, 32, 32)  # (batch, channel, H, W)
        >>> out = net(x)  # (32, 17)
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        output_dim: int = 17,
        conv_channels: List[int] = [32, 64, 64],
        kernel_sizes: List[int] = [3, 3, 3],
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        # Build conv layers
        conv_layers = []
        prev_channels = input_channels
        
        for channels, kernel in zip(conv_channels, kernel_sizes):
            conv_layers.extend([
                nn.Conv2d(prev_channels, channels, kernel, stride=1, padding=kernel//2),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ])
            prev_channels = channels
        
        self.conv = nn.Sequential(*conv_layers)
        
        # Calculate conv output size
        # After 3 max pools on 32x32: 32 -> 16 -> 8 -> 4 = 4x4
        conv_output_size = prev_channels * 4 * 4
        
        # FC layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add channel dim if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.fc(self.conv(x))


class AttentionNetwork(nn.Module):
    """
    Transformer-based network for density grid observations.
    
    Treats the density grid as a sequence of patches and uses
    self-attention to process spatial relationships.
    
    Args:
        grid_size: Size of density grid (assumes square)
        patch_size: Size of each patch
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        output_dim: Output dimension
    
    Example:
        >>> net = AttentionNetwork(32, 4, 128, 4, 2, 17)
        >>> x = torch.randn(32, 32, 32)  # (batch, H, W)
        >>> out = net(x)  # (32, 17)
    """
    
    def __init__(
        self,
        grid_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        output_dim: int = 17,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.num_patches = (grid_size // patch_size) ** 2
        patch_dim = patch_size * patch_size
        
        # Patch embedding
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, H, W) or (batch, 1, H, W)
        if x.dim() == 4:
            x = x.squeeze(1)
        
        batch_size = x.size(0)
        
        # Extract patches: (batch, num_patches, patch_dim)
        patches = x.unfold(1, self.patch_size, self.patch_size)
        patches = patches.unfold(2, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, self.num_patches, -1)
        
        # Embed patches
        x = self.patch_embed(patches)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Use CLS token for output
        return self.head(x[:, 0])


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO/A2C.
    
    Supports shared or separate feature extractors for policy and value.
    
    Args:
        observation_shape: Shape of observations
        action_dim: Dimension of action space
        network_type: 'mlp', 'cnn', or 'attention'
        shared_features: Share feature extractor between actor/critic
        hidden_dims: Hidden dimensions
    
    Example:
        >>> ac = ActorCritic((32, 32), 17, network_type='cnn')
        >>> obs = torch.randn(32, 32, 32)
        >>> action_mean, action_std, value = ac(obs)
    """
    
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        action_dim: int,
        network_type: str = 'mlp',
        shared_features: bool = True,
        hidden_dims: List[int] = [256, 256],
        feature_dim: int = 256,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.shared_features = shared_features
        
        # Create feature extractor
        self.features = self._create_features(
            observation_shape, feature_dim, network_type, hidden_dims
        )
        
        if not shared_features:
            self.critic_features = self._create_features(
                observation_shape, feature_dim, network_type, hidden_dims
            )
        
        # Policy head (actor)
        self.policy_mean = nn.Linear(feature_dim, action_dim)
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Value head (critic)
        self.value = nn.Linear(feature_dim, 1)
        
        self._init_heads()
    
    def _create_features(
        self,
        observation_shape: Tuple[int, ...],
        feature_dim: int,
        network_type: str,
        hidden_dims: List[int],
    ) -> nn.Module:
        """Create feature extractor network."""
        if network_type == 'mlp':
            input_dim = int(torch.prod(torch.tensor(observation_shape)))
            return nn.Sequential(
                nn.Flatten(),
                MLPNetwork(input_dim, hidden_dims, feature_dim),
            )
        elif network_type == 'cnn':
            return CNNNetwork(1, feature_dim, [32, 64, 64], [3, 3, 3], 256)
        elif network_type == 'attention':
            return AttentionNetwork(
                observation_shape[0], 4, 128, 4, 2, feature_dim
            )
        else:
            raise ValueError(f"Unknown network type: {network_type}")
    
    def _init_heads(self):
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        nn.init.zeros_(self.policy_mean.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Observations
        
        Returns:
            action_mean: Mean of action distribution
            action_std: Std of action distribution
            value: State value estimate
        """
        # Extract features
        policy_features = self.features(x)
        
        if self.shared_features:
            value_features = policy_features
        else:
            value_features = self.critic_features(x)
        
        # Policy
        action_mean = self.policy_mean(policy_features)
        action_std = self.policy_log_std.exp().expand_as(action_mean)
        
        # Value
        value = self.value(value_features).squeeze(-1)
        
        return action_mean, action_std, value
    
    def get_action(
        self, 
        x: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            x: Observations
            deterministic: Return mean action if True
        
        Returns:
            action: Sampled or deterministic action
            log_prob: Log probability of action
            value: State value estimate
        """
        action_mean, action_std, value = self(x)
        
        if deterministic:
            action = action_mean
            log_prob = torch.zeros(x.size(0), device=x.device)
        else:
            # Sample from normal distribution
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Clamp action to [-1, 1]
        action = torch.tanh(action)
        
        return action, log_prob, value


def create_network(
    network_type: str,
    observation_shape: Tuple[int, ...],
    output_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function for creating networks.
    
    Args:
        network_type: 'mlp', 'cnn', 'attention', or 'actor_critic'
        observation_shape: Shape of observations
        output_dim: Output dimension
        **kwargs: Additional network arguments
    
    Returns:
        Configured network module
    """
    if network_type == 'mlp':
        input_dim = int(torch.prod(torch.tensor(observation_shape)))
        hidden_dims = kwargs.get('hidden_dims', [256, 256])
        return MLPNetwork(input_dim, hidden_dims, output_dim)
    
    elif network_type == 'cnn':
        return CNNNetwork(1, output_dim, **kwargs)
    
    elif network_type == 'attention':
        return AttentionNetwork(
            observation_shape[0], 
            kwargs.get('patch_size', 4),
            kwargs.get('embed_dim', 128),
            kwargs.get('num_heads', 4),
            kwargs.get('num_layers', 2),
            output_dim
        )
    
    elif network_type == 'actor_critic':
        action_dim = kwargs.get('action_dim', output_dim)
        return ActorCritic(
            observation_shape, 
            action_dim,
            kwargs.get('backbone', 'cnn'),
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown network type: {network_type}")
