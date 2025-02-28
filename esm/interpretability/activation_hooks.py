import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any, Union
import numpy as np
from collections import defaultdict


class ActivationCaptureHook:
    """
    Hook to capture activations from model layers.
    
    Args:
        storage (dict): Dictionary to store captured activations
        key (str): Key to use for storing the activations
        post_process_fn (callable, optional): Function to post-process activations
    """
    def __init__(
        self, 
        storage: Dict[str, Any], 
        key: str, 
        post_process_fn: Optional[Callable] = None
    ):
        self.storage = storage
        self.key = key
        self.post_process_fn = post_process_fn
        
    def __call__(self, module, input_args, output):
        """Capture the output of the module."""
        if isinstance(output, tuple):
            # Some modules return multiple outputs, take the first one
            activation = output[0]
        else:
            activation = output
            
        # Apply post-processing if provided
        if self.post_process_fn is not None:
            activation = self.post_process_fn(activation)
            
        self.storage[self.key] = activation


class ESMCActivationCapturer:
    """
    Utility for capturing activations from ESMC model.
    
    Args:
        model: ESMC model instance
        layers_to_capture: List of layer indices to capture 
        component: Which component to capture ('attention', 'ffn', 'mlp')
    """
    def __init__(
        self,
        model,
        layers_to_capture: List[int] = None,
        component: str = 'mlp'  # 'attention', 'mlp', 'embeddings'
    ):
        self.model = model
        self.component = component
        
        if layers_to_capture is None:
            # Capture all layers by default
            layers_to_capture = list(range(len(model.transformer.layers)))
        self.layers_to_capture = layers_to_capture
        
        self.activations = {}
        self.hooks = []
        
    def attach_hooks(self):
        """Attach hooks to the model to capture activations."""
        # Clear any existing hooks
        self.remove_hooks()
        
        # Different hook points based on the component we want to capture
        if self.component == 'mlp':
            # Capture MLP outputs
            for layer_idx in self.layers_to_capture:
                layer = self.model.transformer.layers[layer_idx]
                hook = layer.mlp.register_forward_hook(
                    ActivationCaptureHook(self.activations, f"layer_{layer_idx}_mlp")
                )
                self.hooks.append(hook)
                
        elif self.component == 'attention':
            # Capture attention outputs
            for layer_idx in self.layers_to_capture:
                layer = self.model.transformer.layers[layer_idx]
                hook = layer.attention.register_forward_hook(
                    ActivationCaptureHook(self.activations, f"layer_{layer_idx}_attention")
                )
                self.hooks.append(hook)
                
        elif self.component == 'embeddings':
            # Capture embeddings output
            hook = self.model.embeddings.register_forward_hook(
                ActivationCaptureHook(self.activations, "embeddings_output")
            )
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def get_activations(self):
        """Get the captured activations."""
        return self.activations
    
    def clear_activations(self):
        """Clear the stored activations."""
        self.activations = {} 