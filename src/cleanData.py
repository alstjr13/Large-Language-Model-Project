import pandas as pd
import re

df = pd.read_csv('sample_6k_reviews_for_RA.csv')

# Distinguish each columns
reviewText = df['reviewText']
incentivized_label = df['incentivized_999']

# Function to remove the sentence with ("free" or "discount" or "reduced") in df["reviewText"]
def remove_incentivized_sentence(text):
    # Add more patterns if needed
    patterns = [
        r'[^\*.(,!?]*\b(free product|free|discount|discounted rate|reduced|reduced price|Disclaimer|\*)\b[^.,!)?]*[.!?]'
    ]
    for pattern in patterns:
        if isinstance(text, str):
            return re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
        else:
            return ''



if __name__ == "__main__":
    # Create another column "cleanedReviewText" with incentivized sentence removed
    df['cleanedReviewText'] = df['reviewText'].apply(remove_incentivized_sentence)

    # Create an another .csv file: cleanedReviewWithLabels.csv
    cleaned_review_df = df[['cleanedReviewText', 'incentivized_999']]
    cleaned_review_df.to_csv('./cleaned_reviews_with_labels.csv', index=False)