import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Define a custom dataset for stance detection
class StanceDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Function to prepare SemEval data with support for longer sequences
def prepare_semeval_data(data_path, tokenizer, max_length=384):
    """
    Prepares the SemEval dataset for fine-tuning.
    Max length increased to 384 to handle complex arguments.
    """
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Map stance labels to integers
    stance_mapping = {'AGAINST': 0, 'FAVOR': 1, 'NONE': 2}
    df['stance_id'] = df['stance'].map(stance_mapping)
    
    # Create input texts: format to preserve the full context
    texts = [f"Topic: {target} Argument: {tweet}" for target, tweet in zip(df['target'], df['tweet'])]
    labels = df['stance_id'].values
    
    # Split into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )
    
    # Tokenize inputs with appropriate max_length for complex arguments
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    
    # Create datasets
    train_dataset = StanceDataset(train_encodings, train_labels)
    val_dataset = StanceDataset(val_encodings, val_labels) 
    
    return train_dataset, val_dataset

# Compute metrics function
def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for the model.
    """
    predictions, labels = eval_pred
    
    # Ensure predictions are in the correct format
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = (predictions == labels).mean()

    # Calculate F1 score per class
    from sklearn.metrics import f1_score, precision_score, recall_score
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_per_class = f1_score(labels, predictions, average=None)
    
    # Create a dictionary with comprehensive metrics
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
    }
    
    # Add class-specific F1 scores
    class_names = ['against', 'favor', 'neutral']
    for i, class_name in enumerate(class_names):
        if i < len(f1_per_class):
            metrics[f'f1_{class_name}'] = f1_per_class[i]
    
    return metrics

# Fine-tuning function with parameters optimized for complex argument handling
def finetune_deberta_for_complex_stance(
    semeval_csv_path, 
    output_dir, 
    model_name="microsoft/deberta-v3-large",  # Using large model for complex arguments
    epochs=4,                                 # Moderate number of epochs 
    use_fp16=True                             # Use mixed precision
):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Prepare data with longer sequence length
    train_dataset, val_dataset = prepare_semeval_data(
        semeval_csv_path, 
        tokenizer, 
        max_length=384  # Increased to handle complex arguments
    )
    
    # Define optimized training arguments balanced for performance and speed
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,        # Small batch size for large model with long sequences
        per_device_eval_batch_size=4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs', 
        logging_steps=50,
        evaluation_strategy="steps",          # Evaluate periodically
        eval_steps=200,                       # Less frequent evaluation to speed up training
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        learning_rate=1e-5,                   # Lower learning rate for better convergence on complex data
        fp16=use_fp16,                        # Mixed precision training
        gradient_accumulation_steps=16,       # Higher accumulation to simulate larger batch sizes
        report_to="tensorboard",
        save_total_limit=2,                   # Keep last 2 checkpoints
        dataloader_num_workers=2,             # Some parallelization in data loading
        group_by_length=True,                 # Group similar length sequences
    )
    
    # Define trainer with compute_metrics
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]  # More patient early stopping
    )
    
    # Train from scratch
    trainer.train()
    
    # Save model
    model.save_pretrained(f"{output_dir}/final-model")
    tokenizer.save_pretrained(f"{output_dir}/final-model")
    
    print(f"Model successfully fine-tuned and saved to {output_dir}/final-model")
    
    return model, tokenizer

# Function to test the fine-tuned model on complex arguments
def test_finetuned_model(model_path, test_examples):
    """
    Test the fine-tuned model on example inputs.
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    stance_labels = {0: "AGAINST", 1: "FAVOR", 2: "NONE"}
    
    for topic, argument in test_examples:
        # Format input - using the training format
        input_text = f"Topic: {topic} Argument: {argument}"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=384)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
            predicted_class = torch.argmax(probs).item()
            
        print(f"Topic: {topic}")
        print(f"Argument: {argument}")
        print(f"Predicted stance: {stance_labels[predicted_class]}")
        print(f"Confidence: {probs[predicted_class].item():.4f}")
        print(f"All probabilities: {dict(zip(stance_labels.values(), probs.tolist()))}")
        print("-" * 50)

# Technique to speed up training while maintaining quality for complex arguments
def train_with_curriculum_learning(
    semeval_csv_path, 
    output_dir,
    model_name="microsoft/deberta-v3-large"
):
    """
    Implements curriculum learning strategy:
    1. First train on shorter sequences (faster)
    2. Then fine-tune further on longer sequences (better quality)
    
    This approach can reduce total training time while maintaining quality.
    """
    # Phase 1: Train on shorter sequences first
    phase1_dir = f"{output_dir}/phase1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare shorter sequence data (faster training)
    df = pd.read_csv(semeval_csv_path)
    
    # Training arguments for phase 1 (shorter, faster)
    training_args_phase1 = TrainingArguments(
        output_dir=phase1_dir,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        fp16=True,
        gradient_accumulation_steps=8,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Run phase 1 training with 180 token limit
    print("Starting Phase 1: Training on shorter sequences")
    model1, _ = finetune_deberta_for_complex_stance(
        semeval_csv_path,
        phase1_dir,
        model_name=model_name,
        epochs=2
    )
    
    # Phase 2: Continue training with longer sequences
    phase2_dir = f"{output_dir}/phase2"
    
    print("Starting Phase 2: Fine-tuning on full-length sequences")
    model2, tokenizer2 = finetune_deberta_for_complex_stance(
        semeval_csv_path,
        phase2_dir,
        model_name=f"{phase1_dir}/final-model",  # Use the phase 1 model as starting point
        epochs=2
    )
    
    # Copy final model to main output directory
    os.makedirs(f"{output_dir}/final-model", exist_ok=True)
    model2.save_pretrained(f"{output_dir}/final-model")
    tokenizer2.save_pretrained(f"{output_dir}/final-model")
    
    print(f"Curriculum learning complete. Final model saved to {output_dir}/final-model")
    
    return model2, tokenizer2

# Example usage
if __name__ == "__main__":
    # Update these paths for your environment
    semeval_data_path = "finetuning/datasets/IBM_SemEval_Merged.csv"
    output_directory = "./stance_ft_deberta"
    
    # Option 1: Standard training with parameters optimized for complex arguments
    model, tokenizer = finetune_deberta_for_complex_stance(
        semeval_csv_path=semeval_data_path,
        output_dir=output_directory,
        model_name="microsoft/deberta-v3-large",
        epochs=3
    )
    
    # Option 2: Curriculum learning approach (potentially faster while maintaining quality)
    # Uncomment to use this approach instead
    # model, tokenizer = train_with_curriculum_learning(
    #     semeval_data_path,
    #     output_directory,
    #     model_name="microsoft/deberta-v3-large"
    # )
    
    # Test examples with complex arguments
    test_examples = [
        ("Climate Change", "While there are some natural climate cycles that have occurred throughout Earth's history, the current rate of warming is unprecedented in the geological record. Multiple independent lines of evidence point to human activity as the primary driver, including carbon isotope analysis, satellite measurements of outgoing radiation, and the observed pattern of warming. The scientific consensus is overwhelming and based on decades of research, not political agendas."),
        ("Abortion", "The ethical complexity of abortion stems from balancing bodily autonomy against the moral status of developing human life. This requires careful consideration of when personhood begins, whether potential personhood confers rights, and how to weigh competing rights claims. Different philosophical traditions and religious beliefs offer varied perspectives, but public policy must navigate these differences while respecting diverse viewpoints in a pluralistic society.")
    ]
    
    test_finetuned_model(f"{output_directory}/final-model", test_examples)