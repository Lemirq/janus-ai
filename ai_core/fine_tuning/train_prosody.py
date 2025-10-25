"""
Fine-tuning script for training Higgs model with prosody tokens
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import json

from prepare_dataset import ProsodyDataset


@dataclass
class ProsodyTrainingConfig:
    """Configuration for prosody fine-tuning"""
    model_name: str = "meta-llama/Llama-3.2-3B"
    output_dir: str = "./prosody_finetuned_model"
    dataset_path: str = "./prosody_training_data.json"
    
    # Training parameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500
    
    # Prosody-specific parameters
    prosody_loss_weight: float = 0.3
    audio_alignment_weight: float = 0.2
    max_prosody_tokens: int = 20
    

class ProsodyAwareModel(nn.Module):
    """
    Wrapper around Higgs/Llama model to add prosody-aware training.
    This extends the DualFFN pathway to handle prosody tokens.
    """
    
    def __init__(self, base_model, config: ProsodyTrainingConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        # Additional layers for prosody processing
        hidden_size = base_model.config.hidden_size
        
        # Prosody token embedding (for tokens >= 128000)
        self.prosody_embeddings = nn.Embedding(
            num_embeddings=100,  # Support 100 prosody tokens
            embedding_dim=hidden_size
        )
        
        # Prosody-to-audio alignment network
        self.prosody_audio_alignment = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 256),  # Audio feature dimension
            nn.Tanh()
        )
        
        # Prosody prediction head (predicts next prosody effect)
        self.prosody_prediction = nn.Linear(hidden_size, 100)
        
    def forward(self, 
                input_ids, 
                attention_mask=None,
                audio_features=None,
                prosody_alignment=None,
                labels=None):
        
        # Separate regular tokens from prosody tokens
        prosody_mask = input_ids >= 128000
        
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = outputs.hidden_states[-1]
        
        # Process prosody tokens
        if prosody_mask.any():
            prosody_indices = torch.where(prosody_mask)
            prosody_token_ids = input_ids[prosody_indices] - 128000
            
            # Get prosody embeddings
            prosody_embeds = self.prosody_embeddings(prosody_token_ids)
            
            # Inject prosody information into hidden states
            hidden_states[prosody_indices] = hidden_states[prosody_indices] + prosody_embeds
            
        # Calculate losses
        total_loss = 0
        
        # Standard language modeling loss
        if labels is not None:
            lm_loss = outputs.loss
            total_loss += lm_loss
            
        # Prosody-audio alignment loss
        if audio_features is not None and prosody_alignment is not None:
            alignment_loss = self._calculate_alignment_loss(
                hidden_states, audio_features, prosody_alignment
            )
            total_loss += self.config.audio_alignment_weight * alignment_loss
            
        # Prosody prediction loss (predict prosody effects)
        if prosody_mask.any():
            prosody_pred_loss = self._calculate_prosody_prediction_loss(
                hidden_states, input_ids, prosody_mask
            )
            total_loss += self.config.prosody_loss_weight * prosody_pred_loss
            
        return {
            'loss': total_loss,
            'logits': outputs.logits,
            'hidden_states': hidden_states
        }
        
    def _calculate_alignment_loss(self, 
                                 hidden_states, 
                                 audio_features, 
                                 prosody_alignment):
        """Calculate loss for aligning prosody tokens with audio regions"""
        
        # Get prosody token positions
        prosody_positions = prosody_alignment['prosody_token_positions']
        
        if not prosody_positions:
            return torch.tensor(0.0, device=hidden_states.device)
            
        # Extract hidden states at prosody positions
        prosody_hidden = hidden_states[:, prosody_positions, :]
        
        # Project to audio space
        predicted_audio = self.prosody_audio_alignment(prosody_hidden)
        
        # Compare with actual audio features at prosody regions
        # This is simplified - real implementation would be more sophisticated
        target_audio = audio_features[:, :predicted_audio.size(1), :predicted_audio.size(2)]
        
        # MSE loss
        loss = nn.functional.mse_loss(predicted_audio, target_audio)
        
        return loss
        
    def _calculate_prosody_prediction_loss(self, 
                                          hidden_states, 
                                          input_ids,
                                          prosody_mask):
        """Calculate loss for predicting prosody effects"""
        
        # Find positions before prosody tokens
        prosody_positions = torch.where(prosody_mask)[1]
        
        if len(prosody_positions) == 0:
            return torch.tensor(0.0, device=hidden_states.device)
            
        # Get hidden states before prosody tokens
        pre_prosody_positions = prosody_positions - 1
        pre_prosody_hidden = hidden_states[:, pre_prosody_positions, :]
        
        # Predict prosody token
        prosody_logits = self.prosody_prediction(pre_prosody_hidden)
        
        # Get actual prosody tokens (shifted by 128000)
        target_prosody = input_ids[:, prosody_positions] - 128000
        
        # Cross entropy loss
        loss = nn.functional.cross_entropy(
            prosody_logits.view(-1, prosody_logits.size(-1)),
            target_prosody.view(-1)
        )
        
        return loss


class ProsodyTrainer(Trainer):
    """Custom trainer for prosody-aware training"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute combined loss for prosody training"""
        
        # Extract all inputs
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        audio_features = inputs.get("audio_features", None)
        prosody_alignment = inputs.get("prosody_alignment", None)
        
        # Shift labels for language modeling
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # Ignore last token
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_features=audio_features,
            prosody_alignment=prosody_alignment,
            labels=labels
        )
        
        loss = outputs['loss']
        
        return (loss, outputs) if return_outputs else loss


def train_prosody_model(config: ProsodyTrainingConfig):
    """Main training function"""
    
    print(f"Loading base model: {config.model_name}")
    
    # Load base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Extend tokenizer vocabulary for prosody tokens
    prosody_tokens = [f"<PROSODY_{i}>" for i in range(100)]
    tokenizer.add_tokens(prosody_tokens, special_tokens=True)
    
    # Resize model embeddings
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Create prosody-aware model
    model = ProsodyAwareModel(base_model, config)
    
    print("Loading dataset...")
    
    # Load datasets
    train_dataset = ProsodyDataset(config.dataset_path)
    eval_dataset = ProsodyDataset(config.dataset_path)  # Would be separate in practice
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        warmup_steps=config.warmup_steps,
        learning_rate=config.learning_rate,
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=True,
        gradient_checkpointing=True,
        report_to=["tensorboard"],
    )
    
    # Create trainer
    trainer = ProsodyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    print("Starting training...")
    
    # Train
    trainer.train()
    
    # Save model
    print(f"Saving model to {config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    # Save prosody token mapping
    prosody_mapping = {
        f"<PROSODY_{i}>": 128000 + i for i in range(100)
    }
    
    with open(f"{config.output_dir}/prosody_tokens.json", "w") as f:
        json.dump(prosody_mapping, f, indent=2)
        
    print("Training complete!")
    

def evaluate_prosody_model(model_path: str, test_examples: List[Dict]):
    """Evaluate the fine-tuned prosody model"""
    
    # Load model
    model = ProsodyAwareModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for example in test_examples:
            # Tokenize input
            inputs = tokenizer(
                example['prosody_text'],
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.8,
                do_sample=True
            )
            
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            results.append({
                'input': example['prosody_text'],
                'generated': generated_text,
                'has_prosody': any(token >= 128000 for token in outputs[0])
            })
            
    return results


if __name__ == "__main__":
    # Example training
    config = ProsodyTrainingConfig(
        model_name="meta-llama/Llama-3.2-3B",
        output_dir="./janus_prosody_model",
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Small for demo
        learning_rate=5e-5
    )
    
    # Uncomment to run training
    # train_prosody_model(config)
    
    print("Prosody training script ready!")
    print(f"Model will be saved to: {config.output_dir}")
    print(f"Training epochs: {config.num_train_epochs}")
    print(f"Batch size: {config.per_device_train_batch_size}")
