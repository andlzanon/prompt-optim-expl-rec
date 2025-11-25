import pandas as pd
from sklearn.model_selection import train_test_split

old_train_path = "../train_test_oficial/train.dat"
old_test_path = "../train_test_oficial/test.dat"

new_train_path = "../train_validation_test_oficial/train.dat"
new_validation_path = "../train_validation_test_oficial/validation.dat"
new_test_path = "../train_validation_test_oficial/test.dat"


old_train_df = pd.read_csv(old_train_path, sep=",", names=["userID", "itemID", "rating", "timestamp"])
old_test_df = pd.read_csv(old_test_path, sep=",", names=["userID", "itemID", "rating", "timestamp"])

new_train_df, new_validation_df = train_test_split(old_train_df, test_size=0.1, random_state=42)
new_test_df = old_test_df

new_train_df = new_train_df.sort_values(["userID", "itemID"])
new_validation_df = new_validation_df.sort_values(["userID", "itemID"])
new_test_df = new_test_df.sort_values(["userID", "itemID"])

print(new_train_df.shape)
print(new_validation_df.shape)
print(new_test_df.shape)

new_train_df.to_csv(new_train_path, index=False, sep=",", header=False)
new_validation_df.to_csv(new_validation_path, index=False, sep=",", header=False)
new_test_df.to_csv(new_test_path, index=False, sep=",", header=False)