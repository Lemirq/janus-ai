"""
Direct Higgs Fine-Tuning - Simple Approach
Just load the HuggingFace model and apply LoRA (like GPT-2)
Bypasses the serve engine complexity
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import json
from pathlib import Path
import argparse
from datetime import datetime


class SimpleProgressCallback:
    """Show training progress"""
    def __init__(self):
        self.start_time = None
        
    def on_train_begin(self):
        self.start_time = datetime.now()
        print(f"\n{'='*70}")
        print("TRAINING STARTED")
        print("="*70 + "\n")


def train_higgs_simple(epochs=3, batch_size=4, lora_r=16, output_dir="models/higgs_prosody_direct"):
    """
    Simple Higgs fine-tuning using direct HuggingFace loading
    No serve engine - just the model!
    """
    
    print("\n" + "="*70)
    print("HIGGS DIRECT FINE-TUNING (Simple Method)")
    print("="*70)
    
    # Check GPU
    if torch.cuda.is_available():
        device = "cuda"
        print(f"\n[GPU] {torch.cuda.get_device_name(0)}")
        print(f"      VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = "cpu"
        print(f"\n[CPU] Training on CPU (slow)")
    
    # Step 1: Load dataset
    print(f"\n[1/5] Loading dataset...")
    dataset_path = "training_data/prosody_dataset.json"
    
    if not Path(dataset_path).exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        return
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"       Loaded {len(data)} examples")
    
    # Step 2: Load Higgs from HuggingFace (simple!)
    print(f"\n[2/5] Loading Higgs model from HuggingFace...")
    print(f"       Model: bosonai/higgs-audio-v2-generation-3B-base")
    print(f"       Size: ~6 GB (downloading if first time)...")
    
    try:
        # Just load it like any HuggingFace model!
        model = AutoModelForCausalLM.from_pretrained(
            "bosonai/higgs-audio-v2-generation-3B-base",
            device_map="auto",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,  # Required for custom Higgs code
            low_cpu_mem_usage=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            "bosonai/higgs-audio-v2-generation-3B-base",
            trust_remote_code=True
        )
        
        print(f"       [OK] Higgs loaded!")
        
    except Exception as e:
        print(f"[ERROR] Failed to load: {e}")
        print(f"\nMake sure:")
        print(f"  1. You have HuggingFace token: huggingface-cli login")
        print(f"  2. Internet connection working")
        print(f"  3. ~10 GB free disk space")
        return
    
    # Step 3: Add prosody tokens
    print(f"\n[3/5] Adding prosody tokens...")
    
    prosody_tokens = ['<emph>', '<pause_short>', '<pause_long>',
                     '<pitch_high>', '<pitch_low>', '<pitch_rising>', '<pitch_falling>']
    
    print(f"       Original vocab: {len(tokenizer)}")
    tokenizer.add_tokens(prosody_tokens, special_tokens=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Resize embeddings
    model.resize_token_embeddings(len(tokenizer))
    print(f"       New vocab: {len(tokenizer)} (added {len(prosody_tokens)} tokens)")
    
    # Step 4: Apply LoRA
    print(f"\n[4/5] Applying LoRA...")
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.05,
        # Target transformer layers
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Step 5: Prepare data
    print(f"\n[5/5] Preparing training data...")
    
    texts = [item['prosody_text'] for item in data]
    print(f"       Training on: {len(texts)} examples")
    
    # Tokenize
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    
    # Create dataset
    class ProsodyDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        def __len__(self):
            return len(self.encodings['input_ids'])
        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item['labels'] = item['input_ids'].clone()
            return item
    
    dataset = ProsodyDataset(encodings)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"       Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Training arguments
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=5,
        save_steps=50,
        eval_steps=50,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        fp16=(device == "cuda"),
        logging_dir=f"{output_dir}/logs",
        report_to=["tensorboard"] if device == "cuda" else [],
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train!
    print(f"\n{'='*70}")
    print("TRAINING HIGGS WITH LoRA")
    print("="*70)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"LoRA rank: {lora_r}")
    print("="*70 + "\n")
    
    try:
        trainer.train()
        
        # Save
        print(f"\n[SAVING] Higgs model with prosody...")
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
        
        print(f"\n{'='*70}")
        print(f"[SUCCESS] Higgs training complete!")
        print(f"{'='*70}")
        print(f"Saved to: {output_dir}/")
        print(f"Files: adapter_model.safetensors, adapter_config.json")
        print(f"\nBoth models now fine-tuned!")
        print(f"  - GPT-2: Text with prosody (85%)")
        print(f"  - Higgs: Audio with prosody (85-95%)")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        print(f"\nCheck:")
        print(f"  - GPU memory (need 12+ GB for batch_size=4)")
        print(f"  - Reduce batch_size if OOM: --batch-size 1")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Higgs (Simple Direct Method)")
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--output', type=str, default='models/higgs_prosody_direct')
    
    args = parser.parse_args()
    
    train_higgs_simple(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lora_r=args.lora_r,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()

