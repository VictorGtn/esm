import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Callable
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
from tqdm.auto import tqdm
from collections import defaultdict

from esm.models.esmc import ESMC
from esm.interpretability.sparse_autoencoder import SparseAutoencoder
from esm.interpretability.activation_hooks import ESMCActivationCapturer


class ESMCInterpreter:
    """
    Tools for interpreting ESMC using sparse autoencoders.
    
    Args:
        model: ESMC model to interpret
        hidden_dim: Dimension for sparse features (default: twice the model dimension)
        device: Device to use
        l1_coefficient: L1 sparsity coefficient
    """
    def __init__(
        self, 
        model: ESMC,
        hidden_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        l1_coefficient: float = 1e-3,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        
        # Default hidden dimension to twice the model dimension
        self.model_dim = model.d_model
        self.hidden_dim = hidden_dim or (2 * self.model_dim)
        self.l1_coefficient = l1_coefficient
        
        # Create activation capturer
        self.capturer = ESMCActivationCapturer(model)
        
        # Dictionary to store autoencoders for different layers
        self.autoencoders = {}
        
        # Determine data type from model
        self.dtype = next(model.parameters()).dtype
        
    def create_autoencoder_for_layer(self, layer_idx: int, component: str = 'mlp') -> SparseAutoencoder:
        """
        Create a sparse autoencoder for a specific layer.
        
        Args:
            layer_idx: Index of the transformer layer
            component: Which component to analyze ('attention', 'mlp', 'embeddings')
            
        Returns:
            SparseAutoencoder instance
        """
        key = f"{component}_{layer_idx}"
        
        # Get input dimension based on component
        if component == 'embeddings':
            input_dim = self.model_dim
        elif component == 'attention':
            input_dim = self.model_dim
        elif component == 'mlp':
            input_dim = self.model_dim
        else:
            raise ValueError(f"Unknown component: {component}")
        
        # Create autoencoder
        autoencoder = SparseAutoencoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            l1_coefficient=self.l1_coefficient,
            dtype=self.dtype
        ).to(self.device)
        
        self.autoencoders[key] = autoencoder
        return autoencoder
        
    def collect_activations(
        self,
        input_sequences: List[str],
        layer_indices: List[int],
        component: str = 'mlp',
        batch_size: int = 4,
    ) -> Dict[str, torch.Tensor]:
        """
        Collect activations from the model for a dataset of inputs.
        
        Args:
            input_sequences: List of protein sequences
            layer_indices: Which layers to collect activations from
            component: Which component to analyze ('attention', 'mlp', 'embeddings')
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping layer keys to tensors of activations
        """
        # Set up activation capturer
        self.capturer = ESMCActivationCapturer(
            self.model, 
            layers_to_capture=layer_indices,
            component=component
        )
        self.capturer.attach_hooks()
        
        all_activations = defaultdict(list)
        
        # Process inputs in batches
        for i in tqdm(range(0, len(input_sequences), batch_size), desc="Collecting activations"):
            batch = input_sequences[i:i+batch_size]
            
            # Clear previous activations
            self.capturer.clear_activations()
            
            # Tokenize sequences
            batch_tokens = [self.model.tokenizer.encode(seq) for seq in batch]
            max_len = max(len(tokens) for tokens in batch_tokens)
            
            # Pad sequences to the same length
            padded_tokens = [
                tokens + [self.model.tokenizer.pad_token_id] * (max_len - len(tokens))
                for tokens in batch_tokens
            ]
            
            # Create attention mask
            attention_mask = torch.tensor([
                [1] * len(tokens) + [0] * (max_len - len(tokens))
                for tokens in batch_tokens
            ], device=self.device)
            
            # Convert to tensor
            tokens_tensor = torch.tensor(padded_tokens, device=self.device)
            
            # Forward pass with no grad
            with torch.no_grad():
                self.model(tokens_tensor, attention_mask=attention_mask)
                
                # Collect activations
                activations = self.capturer.get_activations()
                for k, v in activations.items():
                    # Extract activations, excluding padding tokens
                    for j, seq_len in enumerate(map(len, batch_tokens)):
                        # Get activations for real tokens (exclude padding)
                        # For ESMC, we include all tokens including special tokens 
                        # since they're meaningful for the protein representation
                        seq_acts = v[j, :seq_len]
                        all_activations[k].append(seq_acts)
        
        # Remove hooks
        self.capturer.remove_hooks()
        
        # Process and concatenate all collected activations
        processed_activations = {}
        for k, act_list in all_activations.items():
            # Concatenate all activations along the first dimension
            concatenated = torch.cat([act.reshape(-1, act.size(-1)) for act in act_list], dim=0)
            processed_activations[k] = concatenated
            
        return processed_activations
    
    def train_layer_autoencoder(
        self,
        layer_idx: int,
        activations: torch.Tensor,
        component: str = 'mlp',
        epochs: int = 10,
        batch_size: int = 256,
        lr: float = 1e-3,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train an autoencoder for a specific layer.
        
        Args:
            layer_idx: Layer index
            activations: Tensor of activations [n_samples, hidden_dim]
            component: Component type ('attention', 'mlp', 'embeddings')
            epochs: Number of training epochs
            batch_size: Training batch size
            lr: Learning rate
            save_path: Path to save the trained autoencoder
            
        Returns:
            Dictionary of training metrics
        """
        key = f"{component}_{layer_idx}"
        
        # Create autoencoder if it doesn't exist
        if key not in self.autoencoders:
            self.create_autoencoder_for_layer(layer_idx, component)
        
        autoencoder = self.autoencoders[key]
        optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=lr)
        
        # Training loop
        metrics = defaultdict(list)
        dataset = torch.utils.data.TensorDataset(activations)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        for epoch in range(epochs):
            epoch_metrics = defaultdict(float)
            n_batches = 0
            
            for (x,) in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                x = x.to(self.device)
                
                # Forward pass
                reconstructed, sparse_code = autoencoder(x)
                
                # Compute loss
                loss_dict = autoencoder.loss(x, reconstructed, sparse_code)
                loss = loss_dict["total"]
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track metrics
                for k, v in loss_dict.items():
                    epoch_metrics[k] += v.item() if torch.is_tensor(v) else v
                n_batches += 1
            
            # Compute epoch averages
            for k, v in epoch_metrics.items():
                avg_v = v / n_batches
                metrics[k].append(avg_v)
                
            print(f"Epoch {epoch+1}: recon_loss={metrics['reconstruction'][-1]:.4f}, "
                  f"sparsity={metrics['sparsity'][-1]:.4f}")
        
        # Save model if requested
        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(autoencoder.state_dict(), path)
            
        return dict(metrics)
    
    def interpret_features(
        self,
        layer_idx: int,
        component: str = 'mlp',
        feature_indices: Optional[List[int]] = None,
        top_k: int = 10,
        sample_inputs: Optional[List[str]] = None,
        visualize: bool = True,
    ) -> Dict:
        """
        Interpret the learned features in a trained autoencoder.
        
        Args:
            layer_idx: Layer index
            component: Component type
            feature_indices: Indices of features to interpret (default: top features by importance)
            top_k: Number of top features to interpret
            sample_inputs: Sample sequences to find activating examples
            visualize: Whether to create plots
            
        Returns:
            Dictionary with interpretation results
        """
        key = f"{component}_{layer_idx}"
        
        if key not in self.autoencoders:
            raise ValueError(f"No trained autoencoder for {key}")
            
        autoencoder = self.autoencoders[key]
        
        # Get feature importance
        feature_importance = autoencoder.get_feature_importance()
        
        # Select features to interpret
        if feature_indices is None:
            # Use top-k features by importance
            _, feature_indices = torch.topk(feature_importance, top_k)
            feature_indices = feature_indices.tolist()
        
        # Results dictionary
        results = {
            "feature_importance": feature_importance.cpu().numpy(),
            "interpreted_features": {}
        }
        
        # Analyze decoder weights for each feature
        for idx in feature_indices:
            # Get decoder weights for this feature
            decoder_weights = autoencoder.decoder.weight[:, idx].cpu().detach()
            
            feature_result = {
                "importance": feature_importance[idx].item(),
                "decoder_weights": decoder_weights.numpy(),
            }
            
            results["interpreted_features"][idx] = feature_result
            
            # Find activating examples if sample inputs provided
            if sample_inputs and len(sample_inputs) > 0:
                # Collect activations for these samples
                sample_activations = self.collect_activations(
                    sample_inputs, 
                    layer_indices=[layer_idx],
                    component=component
                )
                
                layer_key = f"layer_{layer_idx}_{component}"
                if layer_key in sample_activations:
                    acts = sample_activations[layer_key]
                    
                    # Find top activating examples
                    top_inputs, top_values = autoencoder.get_top_activating_inputs(
                        acts, idx, top_k=5
                    )
                    
                    feature_result["top_activating_examples"] = {
                        "inputs": top_inputs.cpu().numpy(),
                        "activations": top_values.cpu().numpy()
                    }
        
        # Create visualizations if requested
        if visualize:
            self._visualize_features(results, layer_idx, component)
            
        return results
    
    def _visualize_features(self, results, layer_idx, component):
        """Create visualizations for interpreted features."""
        plt.figure(figsize=(12, 8))
        
        # Plot feature importance
        importance = results["feature_importance"]
        plt.subplot(2, 1, 1)
        plt.bar(range(len(importance)), importance)
        plt.title(f"Feature Importance for {component} Layer {layer_idx}")
        plt.xlabel("Feature Index")
        plt.ylabel("Importance (L2 Norm)")
        
        # Plot decoder weights for top features
        plt.subplot(2, 1, 2)
        for idx, feature_data in list(results["interpreted_features"].items())[:5]:
            plt.plot(feature_data["decoder_weights"], label=f"Feature {idx}")
        
        plt.title("Decoder Weights for Top Features")
        plt.xlabel("Input Dimension")
        plt.ylabel("Weight Value")
        plt.legend()
        plt.tight_layout()
        
        # Save or show the plot
        plt.savefig(f"esmc_sae_{component}_layer{layer_idx}_features.png")
        plt.close() 