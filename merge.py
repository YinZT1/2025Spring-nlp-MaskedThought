from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model and save the result.")
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to the pre-trained base model directory.")
    parser.add_argument("--lora_adapter_path", type=str, required=True,
                        help="Path to the LoRA adapter directory (checkpoint).")
    parser.add_argument("--merged_model_output_path", type=str, required=True,
                        help="Directory path to save the merged model and tokenizer.")
    
    args = parser.parse_args()

    base_model_path = args.base_model_path
    lora_path = args.lora_adapter_path
    merged_model_path = args.merged_model_output_path

    print(f"Starting merge process...")
    print(f"  Base Model Path: {base_model_path}")
    print(f"  LoRA Adapter Path: {lora_path}")
    print(f"  Merged Model Output Path: {merged_model_path}")

    # Load tokenizer from the LoRA adapter path to ensure consistency
    # (e.g., if special tokens like '<mask>' were added during training)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            lora_path,
            local_files_only=True,
            trust_remote_code=True # Added for potentially complex tokenizers
        )
        print(f"Successfully loaded tokenizer from LoRA adapter path: {lora_path}")
    except Exception as e:
        print(f"Warning: Could not load tokenizer from LoRA adapter path ({lora_path}): {e}")
        print(f"Attempting to load tokenizer from base model path: {base_model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                local_files_only=True,
                trust_remote_code=True
            )
            print(f"Successfully loaded tokenizer from base model path: {base_model_path}")
        except Exception as e_base:
            print(f"Error: Could not load tokenizer from base model path either: {e_base}")
            print("Please ensure a valid tokenizer is available at one of these locations.")
            return # Exit if tokenizer cannot be loaded

    # Load base model
    print(f"Loading base model from: {base_model_path}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            local_files_only=True,
            torch_dtype=torch.float16,  # Or bfloat16 if preferred and supported
            device_map="auto" # Automatically distribute model on available devices (e.g., GPU)
        )
        print("Base model loaded successfully.")
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    # Resize token embeddings if necessary (e.g. if new tokens were added for LoRA training)
    # This function was part of your original merge script.
    def resize_model_embeddings(model, tokenizer):
        old_num_tokens = model.get_input_embeddings().weight.shape[0]
        new_num_tokens = len(tokenizer)
        
        if old_num_tokens != new_num_tokens:
            print(f"Resizing model token embeddings from {old_num_tokens} to {new_num_tokens}")
            
            # Store original embedding data before resizing
            input_embeddings_data = model.get_input_embeddings().weight.data.clone()
            output_embeddings_data = model.get_output_embeddings().weight.data.clone()

            model.resize_token_embeddings(new_num_tokens) # This reinitializes new tokens

            # Re-assign old embeddings
            model.get_input_embeddings().weight.data[:old_num_tokens, :] = input_embeddings_data
            model.get_output_embeddings().weight.data[:old_num_tokens, :] = output_embeddings_data
            
            # Initialize new token embeddings (e.g., with average of old ones)
            if new_num_tokens > old_num_tokens:
                input_embeddings_avg = input_embeddings_data.mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings_data.mean(dim=0, keepdim=True)
                model.get_input_embeddings().weight.data[old_num_tokens:] = input_embeddings_avg
                model.get_output_embeddings().weight.data[old_num_tokens:] = output_embeddings_avg
            print("Token embeddings resized.")
        else:
            print("Tokenizer vocabulary size matches model embedding size. No resize needed.")

    resize_model_embeddings(base_model, tokenizer)

    # Load LoRA adapter and merge
    print(f"Loading LoRA adapter from: {lora_path}")
    try:
        model = PeftModel.from_pretrained(
            base_model,
            lora_path,
            local_files_only=True
            # device_map="auto" # PeftModel typically inherits device_map from base_model
        )
        print("LoRA adapter loaded successfully.")
        
        print("Merging LoRA weights into the base model...")
        model = model.merge_and_unload()
        print("Merge complete.")
    except Exception as e:
        print(f"Error loading or merging LoRA adapter: {e}")
        return

    # Save the merged model
    print(f"Saving merged model to: {merged_model_path}")
    os.makedirs(merged_model_path, exist_ok=True)
    try:
        model.save_pretrained(merged_model_path)
        tokenizer.save_pretrained(merged_model_path)
        print(f"Merged model and tokenizer saved successfully to {merged_model_path}")
    except Exception as e:
        print(f"Error saving merged model or tokenizer: {e}")

if __name__ == "__main__":
    main()