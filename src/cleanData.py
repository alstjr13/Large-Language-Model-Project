import pandas as pd


filePath = '../data/sample_6k_reviews_for_RA_updated.csv'
df = pd.read_csv(filePath)

class CleanReview(filePath, df):
    def __init__(self, filePath, df):
        self.filePath = filePath
        self.df = df
    def removeIncent(self, row):
        if row['incentivized_999'] == 1:
            return row['reviewText'].replace(row['incent_bert_highest_score_sent'], '')
        else:
            return row['reviewText']

def removeIncent(row):
    if row['incentivized_999'] == 1:
        return row['reviewText'].replace(row['incent_bert_highest_score_sent'], '')
    else:
        return row['reviewText']

# Distinguish each columns
reviewText = df['reviewText']
incentivized_label = df['incentivized_999']

if __name__ == "__main__":
    # Apply the function removeIncent to all rows of the sample .csv file
    df['reviewText'] = df.apply(removeIncent, axis=1)
    print(df["incentivized_999"].value_counts())

    # Create an another .csv file (cleaned) to feed into the LLM
    df.to_csv('../data/updated_review_sample_for_RA.csv', index=False)
