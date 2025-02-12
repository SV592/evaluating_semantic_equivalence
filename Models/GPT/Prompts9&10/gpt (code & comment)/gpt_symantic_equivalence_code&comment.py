import pandas as pd
import time
from openai import OpenAI

# Create an instance of the OpenAI class
client = OpenAI(
    api_key=""
)


def extract_comments(file_path, sheet_name):
    """
    Extracts the 'input', 'target', and 'prediction' columns from the specified Excel sheet.

    Args:
        file_path (str): Path to the Excel file.
        sheet_name (str): Name of the sheet to extract data from.

    Returns:
        DataFrame: The DataFrame with the original data.
        List[Tuple[str, str, str]]: A list of tuples with (input, target, prediction).
    """
    # Load the Excel file
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Extract the required columns
    comments = list(
        zip(df["input"].dropna(), df["target"].dropna(), df["prediction"].dropna())
    )
    return df, comments


def evaluate_comments_and_append(
    client,
    df,
    comments,
    output_file,
    column_name="gpt-4 code & comment",
    save_interval=10,
):
    """
    Evaluates semantic equivalence of comments using the OpenAI API
    and appends the results to the DataFrame, saving progress at regular intervals.

    Args:
        client (OpenAI): The OpenAI API client.
        df (DataFrame): Original DataFrame.
        comments (List[Tuple[str, str, str]]): A list of (input, target, prediction) tuples.
        output_file (str): Path to save the updated Excel file.
        column_name (str): The column name for GPT responses.
        save_interval (int): The number of rows to process before saving progress.

    Returns:
        None
    """
    responses = []
    row_indices = []  # To track rows with valid comments

    for index, (input_value, target, prediction) in enumerate(comments):
        # Prepare the user_message with input field context
        user_message = (
            f"You are a reviewer analyzing a method-level Java code snippet under review.\n"
            f"Your task is to determine whether the following two reviewers are pointing to the same issue in their feedback:\n\n"
            f"Code Snippet:\n{input_value}\n\n"
            f"Reviewer 1: {target}\n"
            f"Reviewer 2: {prediction}\n"
            f"Focus on whether the intent or meaning of the comments is equivalent.\n"
            f'Write "yes" if both comments are pointing to the same issue or "no" if they are not.\n'
            f"Do not write anything else."
        )

        print(user_message)
        try:
            # Make the API call
            response = client.chat.completions.create(
                model="gpt-4",  # Specify the model
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=50,  # Limit response length
                temperature=0,  # Make response deterministic
            )

            # Extract the response content
            choice = response.choices[0]
            answer = choice.message.content.strip()
            responses.append(answer)
            row_indices.append(index)
            print(f"Processed {index + 1}/{len(comments)}: {answer}")

        except Exception as e:
            print(
                f"An error occurred for input '{input_value}', target '{target}', and prediction '{prediction}': {e}"
            )
            responses.append("Error")
            row_indices.append(index)

        # Pause for 1 second to avoid rate limits
        time.sleep(1)

        # Save progress at regular intervals
        if (index + 1) % save_interval == 0 or (index + 1) == len(comments):
            print(f"Saving progress after {index + 1} rows...")

            # Retrieve or initialize the column for GPT responses
            if column_name not in df:
                df[column_name] = [""] * len(df)  # Initialize column with empty strings

            # Update the DataFrame with the new responses
            for i, response in zip(row_indices, responses):
                df.at[i, column_name] = response

            # Save the updated DataFrame
            df.to_excel(output_file, index=False)
            print(f"Checkpoint saved to {output_file}")

            # Reset temporary storage for next batch
            responses = []
            row_indices = []

    print(f"Final file saved to {output_file}")


if __name__ == "__main__":
    # File paths
    input_file = "manual analysis.xlsx"
    output_file = "manual_analysis_code&comment.xlsx"
    sheet_name = "code_comment"

    # Extract comments from the Excel file
    df, comments = extract_comments(input_file, sheet_name)

    # Evaluate comments with OpenAI API and append results to 'gpt-4 code & comment'
    evaluate_comments_and_append(client, df, comments, output_file, save_interval=10)

    print(f"Updated file saved to {output_file}")
