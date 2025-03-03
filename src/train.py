"""
Training script for fine-tuning Gemma 2 models using LoRA
"""

import os
import argparse
import time
import wandb
from src.data import load_dolly_dataset
from src.model import setup_model, get_callbacks, compile_model, train_model, save_model

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 2 models using LoRA")
    parser.add_argument("--precision", type=str, default="fp32", 
                        choices=["fp32", "mixed_bfloat16", "mixed_float16"],
                        help="Training precision mode")
    parser.add_argument("--lora-rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--max-samples", type=int, default=1000, 
                        help="Maximum number of samples to use")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--save-dir", type=str, default="models", 
                        help="Directory to save models")
    
    args = parser.parse_args()
    
    # Initialize Weights & Biases
    run = wandb.init(
        project="gemma2-lora-finetune",
        name=f"{args.precision}-rank{args.lora_rank}",
        config={
            "precision": args.precision,
            "lora_rank": args.lora_rank,
            "max_samples": args.max_samples,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        }
    )
    
    # Load data
    data = load_dolly_dataset(max_samples=args.max_samples)
    
    # Set up model
    print(f"Setting up model with precision {args.precision} and LoRA rank {args.lora_rank}")
    model = setup_model(precision=args.precision, lora_rank=args.lora_rank)
    
    # Compile model
    model = compile_model(
        model, 
        learning_rate=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    # Get callbacks
    callbacks = get_callbacks()
    
    # Train model
    print(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()
    history, model = train_model(
        model, 
        data, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        callbacks=callbacks
    )
    training_time = time.time() - start_time
    
    # Save model
    model_name = f"gemma2-lora-{args.precision}-rank{args.lora_rank}"
    save_path = os.path.join(args.save_dir, model_name)
    save_model(model, save_path)
    
    # Log training time
    wandb.log({"training_time_seconds": training_time})
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Finish experiment
    wandb.finish()

if __name__ == "__main__":
    main()