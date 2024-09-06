from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import TrainerCallback
import torch
import transformers


model_name = "facebook/opt-30b" ### this probably causes cpu to run out of memory
model_name = "bigscience/bloom-7b1"


def main():
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # Load dataset from the Hugging Face datasets library
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")


    # Tokenize the texts
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)


    tokenized_datasets = dataset.map(tokenize_function, batched=True)


    # Load the data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # TinyLlama uses a causal (not masked) language model, similar to GPT-2
    )

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir='results',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        save_steps=10_000,
        save_total_limit=2,
        fp16=True, 
        deepspeed="ds_config.json",  # Path to DeepSpeed config file        
        gradient_checkpointing=True,
	# report_to='wandb'    
    )
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_name)


    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"]
    )


    print("training")
    # Start the training
    trainer.train()


    # Save the final model and tokenizer
    model.save_pretrained('final_model')
    tokenizer.save_pretrained('final_model')


if __name__ == "__main__":
    main()
