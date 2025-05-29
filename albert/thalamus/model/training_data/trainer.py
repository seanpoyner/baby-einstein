import torch
from pipeline import tokenized_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# Assign pad token: using the EOS token as the pad token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("distilgpt2")

training_args = TrainingArguments(
    output_dir="./thalamus_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_strategy="epoch",  # Save at the end of each epoch.
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
)

# Create a data collator that will dynamically pad the inputs.
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

if __name__ == "__main__":
    trainer.train()         # Run training
    trainer.save_model()    # Save the final model to output_dir.
    tokenizer.save_pretrained("./thalamus_finetuned")  # Save the tokenizer as well
    print("Training complete and model saved!")
    print("Model and tokenizer saved to './thalamus_finetuned'")
