import pandas as pd

def detect_overlap_verbose(train_df, recs_df):
    overlaps = []

    train_grouped = train_df.groupby("user_id")["item_id"].apply(set)
    recs_grouped  = recs_df.groupby("user_id")["item_id"].apply(set)

    for user_id in recs_grouped.index:
        train_items = train_grouped.get(user_id, set())
        rec_items   = recs_grouped[user_id]

        common = train_items.intersection(rec_items)
        for item in common:
            overlaps.append((user_id, item))

    return overlaps

train_PATH = "../datasets/train_test_oficial/train.csv"
recs_PATH = "../datasets/recommendation_files/recommendation_lists/user_knn/params_default/K=200/default_user_knn_K=200_recs.csv"

train_df = pd.read_csv(train_PATH, names=["user_id", "item_id", "rating", "timestamp"])
recs_df = pd.read_csv(recs_PATH, names=["user_id", "item_id", "score"])

overlaps = detect_overlap_verbose(train_df, recs_df)

if overlaps:
    print(f"⚠️ Found {len(overlaps)} leaked recommendations")
    print(overlaps[:10])
else:
    print("✅ Clean recommendations")
