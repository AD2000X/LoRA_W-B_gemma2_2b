"""
Gemma model setup and LoRA fine-tuning functions
"""

import os
import keras
import keras_nlp
from keras.callbacks import EarlyStopping
from wandb.integration.keras import WandbMetricsLogger

def setup_model(precision="fp32", lora_rank=4):
    """Set up Gemma model and enable LoRA"""
    
    # Set precision mode
    if precision == "mixed_float16":
        keras.mixed_precision.set_global_policy('mixed_float16')
    elif precision == "mixed_bfloat16":
        keras.mixed_precision.set_global_policy('mixed_bfloat16')
    else:  # default to fp32
        pass
    
    # Load model
    gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma2_2b_en")
    
    # Enable LoRA
    gemma_lm.backbone.enable_lora(rank=lora_rank)
    
    # Set sequence length limit (to control memory usage)
    gemma_lm.preprocessor.sequence_length = 256
    
    return gemma_lm

def get_callbacks(early_stopping_patience=2):
    """Set up training callbacks"""
    
    early_stopping_cb = EarlyStopping(
        monitor="loss",
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    )
    
    # W&B metrics logger
    wandb_logger = WandbMetricsLogger()
    
    return [early_stopping_cb, wandb_logger]

def compile_model(model, learning_rate=5e-5, weight_decay=0.01):
    """Compile model"""
    
    # Use AdamW optimizer (commonly used for transformer models)
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Exclude layer normalization and bias terms from decay
    optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])
    
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    
    return model

def train_model(model, data, epochs=20, batch_size=4, callbacks=None):
    """Train model"""
    
    history = model.fit(
        data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    return history, model

def save_model(model, save_path):
    """Save model"""
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    return save_path