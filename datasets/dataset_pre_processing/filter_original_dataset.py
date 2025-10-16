import pandas as pd

# transform the original dataset into a filtered one
# turn train and test to [user, item, rating]

train_filtered = pd.read_csv("train_test_original/train_llm.csv")
test_filtered = pd.read_csv("train_test_original/test_llm.csv")

train_filtered = train_filtered[["userId", "movieId", "rating"]]
test_filtered = test_filtered[["userId", "movieId", "rating"]]

train_filtered.columns = ["user", "item", "feedback"]
test_filtered.columns = ["user", "item", "feedback"]

train_filtered.to_csv("../train_test_oficial/train_llm_oficial.csv", sep="\t", index=False, header=False)
test_filtered.to_csv("../train_test_oficial/test_llm_oficial.csv", sep="\t", index=False, header=False)

