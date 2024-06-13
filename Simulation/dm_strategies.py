import numpy as np
import pandas as pd
import json

from copy import deepcopy
import os

import string
import nltk
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# import load_reviews
# from load_reviews import HotelReview
# print("4")
# from load_reviews import reviews

# reviews = HotelReview.reviews

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')

model = SentimentIntensityAnalyzer()


def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)
################################
# CONSTS
################################

REVIEWS = 0
BOT_ACTION = 1
USER_DECISION = 2


################################

def correct_action(information):
    if information["hotel_value"] >= 8:
        return 1
    else:
        return 0

def yes_man(information):
    return 1


def random_action(information):
    return np.random.randint(2)


def user_rational_action(information):
    if information["bot_message"] >= 8:
        return 1
    else:
        return 0


def user_picky(information):
    if information["bot_message"] >= 9:
        return 1
    else:
        return 0


def user_sloppy(information):
    if information["bot_message"] >= 7:
        return 1
    else:
        return 0


def user_short_t4t(information):
    if len(information["previous_rounds"]) == 0 \
            or (information["previous_rounds"][-1][BOT_ACTION] >= 8 and
                information["previous_rounds"][-1][REVIEWS].mean() >= 8) \
            or (information["previous_rounds"][-1][BOT_ACTION] < 8 and
                information["previous_rounds"][-1][REVIEWS].mean() < 8):  # cooperation
        if information["bot_message"] >= 8:  # good hotel
            return 1
        else:
            return 0
    else:
        return 0


def user_picky_short_t4t(information):
    if information["bot_message"] >= 9 or ((information["bot_message"] >= 8) and (
            len(information["previous_rounds"]) == 0 or (
            information["previous_rounds"][-1][REVIEWS].mean() >= 8))):  # good hotel
        return 1
    else:
        return 0


def user_hard_t4t(information):
    if len(information["previous_rounds"]) == 0 \
            or np.min(np.array([((r[BOT_ACTION] >= 8 and r[REVIEWS].mean() >= 8)
                                 or (r[BOT_ACTION] <= 8 and r[REVIEWS].mean() < 8)) for r in
                                information["previous_rounds"]])) == 1:  # cooperation
        if information["bot_message"] >= 8:  # good hotel
            return 1
        else:
            return 0
    else:
        return 0


def history_and_review_quality(history_window, quality_threshold):
    def func(information):
        if len(information["previous_rounds"]) == 0 \
                or history_window == 0 \
                or np.min(np.array([((r[BOT_ACTION] >= 8 and r[REVIEWS].mean() >= 8)
                                     or (r[BOT_ACTION] <= 8 and r[REVIEWS].mean() < 8)) for r in
                                    information["previous_rounds"][
                                    -history_window:]])) == 1:  # cooperation from *result's* perspective
            if information["bot_message"] >= quality_threshold:  # good hotel from user's perspective
                return 1
            else:
                return 0
        else:
            return 0
    return func


def topic_based(positive_topics, negative_topics, quality_threshold):
    def func(information):
        review_personal_score = information["bot_message"]
        for rank, topic in enumerate(positive_topics):
            review_personal_score += int(information["review_features"].loc[topic])*2/(rank+1)
        for rank, topic in enumerate(negative_topics):
            review_personal_score -= int(information["review_features"].loc[topic])*2/(rank+1)
        if review_personal_score >= quality_threshold:  # good hotel from user's perspective
            return 1
        else:
            return 0
    return func


def LLM_based(is_stochastic):
    with open(f"data/baseline_proba2go.txt", 'r') as file:
        proba2go = json.load(file)
        proba2go = {int(k): v for k, v in proba2go.items()}

    if is_stochastic:
        def func(information):
            review_llm_score = proba2go[information["review_id"]]
            return int(np.random.rand() <= review_llm_score)
        return func
    else:
        def func(information):
            review_llm_score = proba2go[information["review_id"]]
            return int(review_llm_score >= 0.5)
        return func

# import time
def sentiment(loaded_model,loaded_tfidf):
    def func(information):
       # start = time.time()
       # print('start sentiment')
       # review = reviews[information['review_id']]
       pos_part = deepcopy(information['positive'])
       neg_part = deepcopy(information['negative'])
       # score = review['score']


       # try:
       #     score = float(score)
       # except ValueError:
       #     pass



       missing_values = ['N/A', 'NA', 'NULL', 'NaN', '_', 'MISSING', 'UNKNOWN', 'UNAVAILABLE']

       if not isinstance(pos_part, str) or pos_part.strip().lower() in missing_values:
           pos_part = ''



       if not isinstance(neg_part, str) or neg_part.strip().lower() in missing_values:
           neg_part = ''



       review = pos_part + " " + neg_part
       sentiment = model.polarity_scores(review)
       cleaned_review = clean_text(review)
       num_chars = len(review)
       num_words = len(review.split(" "))
       input = [sentiment['neg'], sentiment['neu'], sentiment['pos'], sentiment['compound'], num_chars, num_words]

       input_tfidf = loaded_tfidf.transform([cleaned_review]).toarray().flatten()

       input.extend(input_tfidf)

       # Create DataFrame with the same feature names used during training
       feature_names = ['neg', 'neu', 'pos', 'compound', 'num_chars', 'num_words'] + ["word_" + str(x) for x in
                                                                                      loaded_tfidf.get_feature_names_out()]
       input_df = pd.DataFrame([input], columns=feature_names)
       # input = input + list(input_tfidf[0])

       prediction = loaded_model.predict(input_df)[0]
       # print('end sentiment')
       # print(time.time()-start)
       return 1 - prediction
    return func

