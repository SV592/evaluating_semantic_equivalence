import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


def extract_comments(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # Extract the code snippets from the "input" column
    comments = list(df["input"].dropna())
    return df, comments


def generate_codebert_embeddings(
    df, comments, output_file, tokenizer, model=None, column_name="codebert_embeddings"
):
    # Initialize model if not provided
    if model is None:
        model_name = "microsoft/codebert-base"
        model = AutoModel.from_pretrained(model_name)
        model.eval()
    # Initialize column if it doesn't exist
    if column_name not in df:
        df[column_name] = [None] * len(df)

    # Loop through each code snippet
    for idx, code_snippet in enumerate(comments):
        # Tokenize with truncation to avoid length errors
        inputs = tokenizer(
            code_snippet, return_tensors="pt", truncation=True, max_length=512
        )

        # Generate embeddings (inference mode)
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract [CLS] token embedding (position 0)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: [1, hidden_dim]
        # Convert to Python list
        embedding_list = cls_embedding.squeeze().tolist()

        # Assign embedding to the corresponding row
        df.at[idx, column_name] = embedding_list

        # Save progress periodically
        if (idx + 1) % 10 == 0:
            df.to_excel(output_file, index=False)
            print(f"Saved progress at row {idx + 1}")

    # Final save
    df.to_excel(output_file, index=False)
    print(f"Final embeddings saved to {output_file}")


if __name__ == "__main__":
    # Model name for CodeBERT
    model_name = "microsoft/codebert-base"

    # Load tokenizer and model outside the function so they're initialized only once
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # File paths and sheet name
    input_file = "manual analysis.xlsx"
    output_file = "manual_analysis_with_codebert_embeddings.xlsx"
    sheet_name = "code_comment"

    # 1. Extract original DataFrame and code snippets
    df, comments = extract_comments(input_file, sheet_name)
    # 2. Generate and store CodeBERT embeddings
    generate_codebert_embeddings(df, comments, output_file, tokenizer, model)
