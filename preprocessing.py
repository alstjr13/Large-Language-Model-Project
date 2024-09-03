import pandas as pd
from sklearn.model_selection import train_test_split

class ReviewPreprocessor:
    def __init__(self, file_path, sample_size, test_size=0.2, random_state=42):
        self.file_path = file_path
        self.sample_size = sample_size
        self.test_size = test_size
        self.random_state = random_state
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path)

    def sample_reviews(self):
        not_incentivized = self.df[self.df['incentivized_999'] == 0].sample(n=self.sample_size, random_state=self.random_state)
        incentivized = self.df[self.df['incentivized_999'] == 1].sample(n=self.sample_size, random_state=self.random_state)

        combined_df = pd.concat([not_incentivized, incentivized])
        new_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

        return new_df

    def split_data(self, new_df):
        X = new_df["reviewText"]
        y = new_df["incentivized_999"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def preprocess(self):
        self.load_data()
        sample_df = self.sample_reviews()
        self.split_data(sample_df)
