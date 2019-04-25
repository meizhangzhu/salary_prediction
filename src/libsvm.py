import json

with open("../data/dataset.json") as f:
    dataset = json.load(f)

trn_data = dataset["train"]
dev_data = dataset["dev"]

X_trn, Y_trn = zip(*trn_data)
X_dev, Y_dev = zip(*dev_data)

