from sentence_transformers import SentenceTransformer
import pandas as pd

# Load the Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(df, output_file, embedding_column="Embedding"):
    """
    Generates embeddings for the 'target' and 'prediction' columns in the DataFrame
    using Sentence-BERT and saves them as strings in an Excel file.

    Args:
        df (DataFrame): Original DataFrame containing 'target' and 'prediction'.
        output_file (str): Path to save the updated Excel file.
        embedding_column (str): Base column name for embeddings.

    Returns:
        None
    """
    embeddings_target = []
    embeddings_prediction = []

    for index, row in df.iterrows():
        try:
            # Generate embeddings for target and prediction
            target_embedding = model.encode(row["target"]).tolist()  # Convert to list
            prediction_embedding = model.encode(
                row["prediction"]
            ).tolist()  # Convert to list

            # Append embeddings
            embeddings_target.append(target_embedding)
            embeddings_prediction.append(prediction_embedding)

            print(f"Processed row {index + 1}/{len(df)}")

        except Exception as e:
            print(f"Error processing row {index}: {e}")
            embeddings_target.append(None)
            embeddings_prediction.append(None)

    # Add embeddings as strings to the DataFrame
    df[f"{embedding_column}_Target"] = embeddings_target
    df[f"{embedding_column}_Prediction"] = embeddings_prediction

    # Save the updated DataFrame to Excel
    df.to_excel(output_file, index=False)
    print(f"Embeddings saved to {output_file}")


if __name__ == "__main__":
    # File paths
    input_file = "manual analysis.xlsx"
    output_file = "manual_analysis_embeddings.xlsx"
    sheet_name = "code_comment"

    # Load the Excel file
    df = pd.read_excel(input_file, sheet_name=sheet_name)

    # Generate embeddings
    generate_embeddings(df, output_file)

    print(f"Embeddings saved to {output_file}")
