"""
STEP 3: Fine-Tune with LoRA
Train the model to understand prosody tokens
"""

import os
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
import json
import argparse
from datetime import datetime
from pathlib import Path


class ProgressCallback(TrainerCallback):
    """Show detailed progress during training"""
    
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.start_time = None
        self.epoch_start = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.now()
        print(f"\n{'='*70}")
        print(f"TRAINING STARTED")
        print(f"{'='*70}\n")
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = datetime.now()
        epoch = int(state.epoch) + 1 if state.epoch else 1
        print(f"\n{'-'*70}")
        print(f"EPOCH {epoch}/{args.num_train_epochs}")
        print(f"{'-'*70}")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            elapsed = (datetime.now() - self.epoch_start).total_seconds() if self.epoch_start else 0
            
            step = state.global_step
            epoch = int(state.epoch) if state.epoch else 0
            loss = logs.get('loss', 0)
            lr = logs.get('learning_rate', 0)
            
            mins, secs = divmod(int(elapsed), 60)
            
            print(f"  Step {step:3d} | Loss: {loss:.4f} | LR: {lr:.2e} | Time: {mins}m {secs}s")
            
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch else 0
        print(f"\n[COMPLETE] Epoch {epoch} done")
        
    def on_train_end(self, args, state, control, **kwargs):
        total_time = (datetime.now() - self.start_time).total_seconds()
        hours, remainder = divmod(int(total_time), 3600)
        mins, secs = divmod(remainder, 60)
        
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"Total time: {hours}h {mins}m {secs}s")
        
        # Get final loss safely
        if state.log_history:
            final_loss = state.log_history[-1].get('loss', 'N/A')
            if isinstance(final_loss, (int, float)):
                print(f"Final loss: {final_loss:.4f}")
            else:
                print(f"Final loss: {final_loss}")
        
        print("="*70 + "\n")


def setup_lora_model(model_name="gpt2", lora_r=16, lora_alpha=32):
    """Setup model with LoRA"""
    
    print(f"\n{'='*70}")
    print("MODEL SETUP")
    print("="*70)
    
    print(f"\n[1/5] Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"[2/5] Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"[3/5] Adding prosody tokens")
    prosody_tokens = ['<emph>', '<pause_short>', '<pause_long>', 
                      '<pitch_high>', '<pitch_low>', '<pitch_rising>', '<pitch_falling>']
    tokenizer.add_tokens(prosody_tokens, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"      Added {len(prosody_tokens)} tokens")
    print(f"      New vocab size: {len(tokenizer)}")
    
    print(f"[4/5] Configuring LoRA")
    
    # Check if GPU is available
    import torch
    if torch.cuda.is_available():
        print(f"       GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"       VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"       WARNING: No GPU detected! Training will be slow.")
        print(f"       Install GPU PyTorch: pip install -r requirements.txt")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj"],  # GPT-2 attention layers
        bias="none"
    )
    
    print(f"[5/5] Applying LoRA adapters")
    model = get_peft_model(model, lora_config)
    
    # Ensure model is trainable
    model.train()
    
    # Enable gradients for LoRA parameters
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True
    
    # Calculate trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"\n{'='*70}")
    print(f"Model Ready:")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable (LoRA): {trainable:,}")
    print(f"  Percentage: {100 * trainable / total:.2f}%")
    print(f"  LoRA rank: {lora_r}")
    print("="*70)
    
    return model, tokenizer


def load_dataset(data_path="training_data/prosody_dataset.json"):
    """Load prepared dataset"""
    
    print(f"\nLoading dataset: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"[OK] Loaded {len(data)} examples")
    
    return data


def train(epochs=3, batch_size=4, lora_r=16, test_mode=False, output_dir="models/janus_prosody_lora"):
    """Main training function"""
    
    # Setup
    model, tokenizer = setup_lora_model(lora_r=lora_r)
    
    # Load data
    dataset_path = "training_data/prosody_dataset.json"
    if not Path(dataset_path).exists():
        print(f"\n[ERROR] Dataset not found: {dataset_path}")
        print("\nRun these first:")
        print("  python 1_segment_audio.py")
        print("  python 2_prepare_data.py")
        return
    
    # Create simple text dataset for testing
    print(f"\nCreating training dataset...")
    data = load_dataset(dataset_path)
    
    # Simple text-only training (for demo - full version would use audio features)
    texts = [ex['text'] for ex in data]
    
    if test_mode:
        texts = texts[:10]
        epochs = 1
        print(f"\n[TEST MODE] Using {len(texts)} examples, 1 epoch")
    
    # Tokenize
    print(f"Tokenizing {len(texts)} examples...")
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    
    # Create dataset with labels
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        def __len__(self):
            return len(self.encodings['input_ids'])
        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            # Labels are the input_ids shifted (for language modeling)
            item['labels'] = item['input_ids'].clone()
            return item
    
    dataset = SimpleDataset(encodings)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"[OK] Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Training arguments
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=3e-4,
        warmup_steps=min(100, len(train_dataset) // batch_size),
        logging_steps=1,  # Log every step
        save_steps=max(50, len(train_dataset) // (batch_size * 2)),
        eval_steps=max(50, len(train_dataset) // (batch_size * 2)),
        eval_strategy="steps",  # Changed from evaluation_strategy (newer transformers)
        save_strategy="steps",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),  # Use FP16 if GPU available
        gradient_checkpointing=False,  # Disable to avoid gradient issues
        logging_dir=f"{output_dir}/logs",
        report_to=["tensorboard"] if torch.cuda.is_available() else [],
        use_cpu=(not torch.cuda.is_available()),  # Explicitly use CPU if no GPU
    )
    
    # Calculate total steps for progress
    steps_per_epoch = len(train_dataset) // batch_size
    total_steps = steps_per_epoch * epochs
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[ProgressCallback(total_steps)]
    )
    
    # Train!
    trainer.train()
    
    # Save
    print(f"\nSaving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save prosody mapping
    prosody_map = {
        '<emph>': 5000000,
        '<pause_short>': 5000001,
        '<pause_long>': 5000002,
        '<pitch_high>': 5000003,
        '<pitch_low>': 5000004,
        '<pitch_rising>': 5000005,
        '<pitch_falling>': 5000006
    }
    
    with open(f"{output_dir}/prosody_tokens.json", 'w') as f:
        json.dump(prosody_map, f, indent=2)
    
    print(f"\n[SAVED] Model saved to: {output_dir}/")
    print(f"[SAVED] Files created:")
    print(f"    - adapter_model.bin")
    print(f"    - adapter_config.json")
    print(f"    - tokenizer/")
    print(f"    - prosody_tokens.json")
    
    print(f"\n[COMPLETE] STEP 3 DONE!")
    print(f"Model ready to use!")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune with LoRA for prosody")
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs (default: 3)')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size (default: 4)')
    parser.add_argument('--lora-r', type=int, default=16, help='LoRA rank (default: 16)')
    parser.add_argument('--test-mode', action='store_true', help='Quick test (10 samples, 1 epoch)')
    parser.add_argument('--output', type=str, default='models/janus_prosody_lora', help='Output directory')
    
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lora_r=args.lora_r,
        test_mode=args.test_mode,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()

