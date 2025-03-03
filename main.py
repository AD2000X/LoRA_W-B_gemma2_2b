"""
Gemma 2 LoRA Fine-tuning Main Program
"""

import os
import argparse
from src.train import main as train_main

def main():
    """Main program entry point"""
    parser = argparse.ArgumentParser(description="Gemma 2 LoRA Fine-tuning Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Fine-tune Gemma 2 model with LoRA")
    train_parser.add_argument("--precision", type=str, default="fp32", 
                        choices=["fp32", "mixed_bfloat16", "mixed_float16"],
                        help="Training precision mode")
    train_parser.add_argument("--lora-rank", type=int, default=4, help="LoRA rank")
    train_parser.add_argument("--max-samples", type=int, default=1000, 
                        help="Maximum number of samples to use")
    train_parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    train_parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    train_parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    train_parser.add_argument("--save-dir", type=str, default="models", 
                        help="Directory to save models")

    # Evaluate subcommand
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    evaluate_parser.add_argument("--model-path", type=str, required=True, 
                          help="Path to trained model")
    evaluate_parser.add_argument("--prompt", type=str, 
                          default="What should I do on a trip to Europe?", 
                          help="Prompt for text generation")
    evaluate_parser.add_argument("--max-length", type=int, default=256, 
                          help="Maximum length of generated text")

    args = parser.parse_args()
    
    if args.command == "train":
        train_main()
    elif args.command == "evaluate":
        from src.evaluate import evaluate_model_generation
        evaluate_model_generation(
            args.model_path, 
            args.prompt, 
            args.max_length
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()