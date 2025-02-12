import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


def generate_embeddings_for_column(df, column_name, tokenizer, model, output_column):
    """
    Generates embeddings for the specified column in df and stores them in output_column.
    """
    # Initialize the output column
    df[output_column] = [None] * len(df)

    for idx, text in df[column_name].items():
        if pd.isna(text):
            continue  # Skip NaN entries

        # Tokenize with truncation
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract [CLS] embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        df.at[idx, output_column] = cls_embedding.squeeze().tolist()

    return df


if __name__ == "__main__":
    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Excel details
    input_file = "../manual analysis.xlsx"
    output_file = "manual_analysis_with_codebert_embeddings.xlsx"
    sheet_name = "code_comment"

    # Load DataFrame
    df = pd.read_excel(input_file, sheet_name=sheet_name)

    # Generate embeddings for the 'target' column
    df = generate_embeddings_for_column(
        df,
        column_name="target",
        tokenizer=tokenizer,
        model=model,
        output_column="codeBERT_comment_target_embeddings",
    )

    # Generate embeddings for the 'prediction' column
    df = generate_embeddings_for_column(
        df,
        column_name="prediction",
        tokenizer=tokenizer,
        model=model,
        output_column="codeBERT_comment_prediction_embeddings",
    )

    # Save the updated DataFrame
    df.to_excel(output_file, index=False)
    print(f"Embeddings added and saved to {output_file}")
