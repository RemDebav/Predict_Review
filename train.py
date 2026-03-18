import pandas as pd
from collections import Counter
import re
import pickle

# ── 1. Chargement ──────────────────────────────────────────────
df = pd.read_csv("IMDBDataset.csv")

# ── 2. Split train/test ────────────────────────────────────────
split = int(0.90 * len(df))
train_df = df[:split]

# ── 3. Comptage des mots ───────────────────────────────────────
def count_words(reviews):
    word_counter = Counter()
    for review in reviews:
        words = re.findall(r'\b\w+\b', review.lower())
        word_counter.update(words)
    return dict(word_counter)

positive_reviews = train_df[train_df['sentiment'] == 'positive']['review']
negative_reviews = train_df[train_df['sentiment'] == 'negative']['review']

positive_words_dict = count_words(positive_reviews)
negative_words_dict = count_words(negative_reviews)

vocab = set(positive_words_dict.keys()) | set(negative_words_dict.keys())

p_positive = len(positive_reviews) / len(train_df)
p_negative = len(negative_reviews) / len(train_df)

# ── 4. Sauvegarde du modèle ────────────────────────────────────
model = {
    "positive_words_dict": positive_words_dict,
    "negative_words_dict": negative_words_dict,
    "vocab_size":          len(vocab),
    "total_pos":           sum(positive_words_dict.values()),
    "total_neg":           sum(negative_words_dict.values()),
    "p_positive":          p_positive,
    "p_negative":          p_negative,
}

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"Entraînement terminé sur {len(train_df)} exemples.")
print(f"  Vocabulaire : {len(vocab)} mots")
print(f"  P(positive) = {p_positive:.2%}  |  P(negative) = {p_negative:.2%}")
print("Modèle sauvegardé dans model.pkl")
