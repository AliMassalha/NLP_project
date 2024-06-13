import os
import pandas as pd

class HotelReview:
    # Class attribute
    reviews = {}

    @classmethod
    def add_review(cls, review_id, positive_review, negative_review, score):
        cls.reviews[review_id] = {"positive": positive_review,
                                  "negative": negative_review,
                                  "score": score}

    @classmethod
    def load_reviews(cls, reviews_path, n_hotels):
        if len(cls.reviews) == 0:
            for h in range(1, n_hotels + 1):
                hotel_df = pd.read_csv(os.path.join(reviews_path, f"{h}.csv"), header=None)
                for _, review in hotel_df.iterrows():
                    review_id = review[0]
                    positive_review = review[2]
                    negative_review = review[3]
                    score = review[4]
                    cls.add_review(review_id, positive_review, negative_review, score)

            # Add a default review
            cls.add_review(-1, "", "", 8)


# Initialize reviews when the module is imported
DATA_GAME_REVIEWS_PATH = "data/game_reviews"
N_HOTELS = 1068

# Load reviews into the class attribute
HotelReview.load_reviews(DATA_GAME_REVIEWS_PATH, N_HOTELS)
# print('hi')
reviews = HotelReview.reviews