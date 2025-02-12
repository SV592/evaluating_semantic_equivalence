import requests
import pandas as pd
import time

# Hugging Face API details
API_URL = (
    "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
)
API_TOKEN = ""  # Replace with your API token
headers = {"Authorization": f"Bearer {API_TOKEN}"}


def query_huggingface_api(source, target_list):
    payload = {"inputs": {"source_sentence": source, "sentences": target_list}}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()  # Returns a list of similarity scores


def evaluate_comments_with_huggingface(
    df,
    output_file,
    score_column="Similarity Score",
    decision_column="Semantic Similarity",
    save_interval=10,
):
    """
    Evaluates semantic equivalence of comments using Hugging Face Inference API
    and appends the similarity scores and equivalence decisions to the DataFrame.

    Args:
        df (DataFrame): Original DataFrame containing 'target' and 'prediction'.
        output_file (str): Path to save the updated Excel file.
        score_column (str): Column name for similarity scores.
        decision_column (str): Column name for equivalence decisions.
        save_interval (int): Number of rows to process before saving progress.

    Returns:
        None
    """
    scores = []
    decisions = []
    row_indices = []

    for index, row in df.iterrows():
        # Extract target and prediction
        target = row["target"]
        prediction = row["prediction"]

        # Query Hugging Face API
        try:
            similarity_scores = query_huggingface_api(target, [prediction])
            similarity_score = similarity_scores[
                0
            ]  # Get the similarity score for the pair

            # Decide equivalence based on a threshold
            equivalence = (
                "Yes" if similarity_score >= 0.85 else "No"
            )  # Adjust threshold as needed

            # Save the results
            scores.append(similarity_score)
            decisions.append(equivalence)
            row_indices.append(index)
            print(
                f"Processed row {index + 1}/{len(df)}: {equivalence} (Score: {similarity_score:.2f})"
            )

        except Exception as e:
            print(f"Error processing row {index}: {e}")
            scores.append("Error")
            decisions.append("Error")
            row_indices.append(index)

        # Pause for 1 second to avoid rate limits
        time.sleep(1)

        # Save progress at regular intervals
        if (index + 1) % save_interval == 0 or (index + 1) == len(df):
            print(f"Saving progress after {index + 1} rows...")

            # Initialize columns if not already present
            if score_column not in df:
                df[score_column] = [""] * len(df)
            if decision_column not in df:
                df[decision_column] = [""] * len(df)

            # Update the DataFrame with the new results
            for i, score, decision in zip(row_indices, scores, decisions):
                df.at[i, score_column] = score
                df.at[i, decision_column] = decision

            # Save the updated DataFrame
            df.to_excel(output_file, index=False)
            print(f"Checkpoint saved to {output_file}")

            # Reset temporary storage for next batch
            scores = []
            decisions = []
            row_indices = []

    print(f"Final file saved to {output_file}")


if __name__ == "__main__":
    # File paths
    input_file = "manual analysis.xlsx"
    output_file = "manual_analysis_huggingface.xlsx"
    sheet_name = "code_comment"

    # Load the Excel file
    df = pd.read_excel(input_file, sheet_name=sheet_name)

    # Evaluate semantic similarity with Hugging Face API
    evaluate_comments_with_huggingface(df, output_file, save_interval=10)

    print(f"Updated file saved to {output_file}")
