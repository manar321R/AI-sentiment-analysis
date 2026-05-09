# 2.6
# Load BERT and Configure Training
# ============================================================
MODEL_NAME = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# Prepare data
train_dataset = SentimentDataset(train_df['clean_text'], train_df['label'], tokenizer)
val_dataset = SentimentDataset(val_df['clean_text'], val_df['label'], tokenizer)


# Fast training settings with fp16 support
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    learning_rate=3e-5,                  # Added here to control learning rate
    per_device_train_batch_size=32,
    fp16=True,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    logging_steps=100,
    report_to="none"
)

# Use standard Trainer directly
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

print('✅ BERT loaded and training config ready')

# 2.7
# Train BERT and Save Best Model
# ============================================================

# Ensure Google Drive is connected
drive.mount('/content/drive')

try:
    # Path to save checkpoints inside Drive to protect against interruption
    CHECKPOINT_DIR = '/content/drive/MyDrive/sentiment_bert_checkpoints'
    # Path to save the best final model
    BEST_MODEL_DIR = '/content/drive/MyDrive/sentiment_bert_model_best'

    # 1. Initialize model and tokenizer
    MODEL_NAME = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    # 2. Build data
    train_dataset = SentimentDataset(train_df['clean_text'], train_df['label'], tokenizer)
    val_dataset = SentimentDataset(val_df['clean_text'], val_df['label'], tokenizer)

    # 3. Training settings (with temporary saving to Drive)
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=3,
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        fp16=True,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        logging_steps=100,
        report_to="none"
    )

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # 5. Search for the last checkpoint
    last_checkpoint = None
    if os.path.exists(CHECKPOINT_DIR):
        last_checkpoint = get_last_checkpoint(CHECKPOINT_DIR)

    if last_checkpoint is not None:
        print(f"🔄 Previous checkpoint found! Resuming training from: {last_checkpoint}")
    else:
        print('🚀 Starting training from scratch...')

    # 6. Start or resume training
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # 7. Save the best version in a separate and clear folder within Drive
    print(f'⏳ Saving the best version of the model permanently to: {BEST_MODEL_DIR}')
    model.save_pretrained(BEST_MODEL_DIR)
    tokenizer.save_pretrained(BEST_MODEL_DIR)

    print('✅ Training completed and best model secured in Drive successfully!')

except Exception as e:
    print(f'❌ An error occurred: {e}')
    import traceback
    traceback.print_exc()
