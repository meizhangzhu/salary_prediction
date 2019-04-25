import code
import json

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# features = ["jobId", "companyId", "jobType", "degree", "major", "industry", "yearsExperience", "milesFromMetropolis"]

xdf = pd.read_csv("../data/train_features.csv")
ydf = pd.read_csv("../data/train_salaries.csv")
test_xdf = pd.read_csv("../data/test_features.csv")
np_df = np.array(xdf)

jobId, companyId, jobType, degree, major, industry, yearsExperience, milesFromMetropolis = np_df.T
feat_dict = {"jobId":jobId, "companyId":companyId, "jobType":jobType, "degree":degree, "major":major, "industry":industry, "yearsExperience":yearsExperience, "milesFromMetropolis":milesFromMetropolis}

sta_y = []
for feat in feat_dict:
    print(feat)
    count = 0
    for item in feat_dict[feat]:
        if item=='NONE':
            count += 1
    print("None sample:{}".format(count))
    sta_y.append((len(feat_dict[feat])-count)/len(feat_dict[feat]))
index = [idx for idx in range(len(feat_dict))]
# plt.bar(index, sta_y)
# plt.show()
# cate_feat = ["companyId", "jobType", "degree", "major", "industry"]
X = xdf[['companyId', 'jobType', 'degree', 'major', 'industry', 'yearsExperience', 'milesFromMetropolis']]
Y = ydf[['salary']]
test_X = test_xdf[['companyId', 'jobType', 'degree', 'major', 'industry', 'yearsExperience', 'milesFromMetropolis']]

# code.interact(local=locals())

# process companyId
labels = X["companyId"].unique().tolist()
def companyIdMap(x):
    return labels.index(x)
X["companyId"] = X["companyId"].apply(companyIdMap)
test_X["companyId"] = test_X["companyId"].apply(companyIdMap)

# process jobType
labels = ['NONE', 'CFO', 'CEO', 'VICE_PRESIDENT', 'MANAGER', 'JUNIOR', 'JANITOR', 'CTO', 'SENIOR']
subset_labels = ['NONE', 'CFO', 'CEO', 'JUNIOR']
def jobTypeMap(x):
    return labels.index(x)
    # if x in subset_labels:
    #     return subset_labels.index(x)
    # else:
    #     return 0
X["jobType"] = X["jobType"].apply(jobTypeMap)
test_X["jobType"] = test_X["jobType"].apply(jobTypeMap)

# process degree
labels = ['NONE', 'MASTERS', 'HIGH_SCHOOL', 'DOCTORAL', 'BACHELORS']
subset_labels = ['NONE', 'DOCTORAL', 'BACHELORS']
def degreeMap(x):
    return labels.index(x)
    # if x in subset_labels:
    #     return subset_labels.index(x)
    # else:
    #     return 0
X["degree"] = X["degree"].apply(degreeMap)
test_X["degree"] = test_X["degree"].apply(degreeMap)

# process major
labels = ['NONE', 'MATH', 'PHYSICS', 'CHEMISTRY', 'COMPSCI', 'BIOLOGY', 'LITERATURE', 'BUSINESS', 'ENGINEERING']
subset_labels = ['NONE', 'BUSINESS', 'ENGINEERING']
def majorMap(x):
    return labels.index(x)
    # if x in subset_labels:
    #     return subset_labels.index(x)
    # else:
    #     return 0
X["major"] = X["major"].apply(majorMap)
test_X["major"] = test_X["major"].apply(majorMap)

# process industry
labels = ['NONE', 'HEALTH', 'WEB', 'AUTO', 'FINANCE', 'EDUCATION', 'OIL', 'SERVICE']
subset_labels = ['NONE', 'HEALTH', 'WEB', 'FINANCE', 'EDUCATION', 'OIL']
def industryMap(x):
    return labels.index(x)
    # if x in subset_labels:
    #     return subset_labels.index(x)
    # else:
    #     return 0
X["industry"] = X["industry"].apply(industryMap)
test_X["industry"] = test_X["industry"].apply(industryMap)

# split for trn and validation.
X_trn, X_dev, Y_trn, Y_dev = train_test_split(X, Y, test_size = 0.20, random_state = 0)

dataset = {"train": [], "dev": [], "test": []}

X_trn_lst = X_trn.values.tolist()
Y_trn_lst = Y_trn.values.tolist()
for x, y in zip(X_trn_lst, Y_trn_lst):
    dataset["train"].append([x, y[0]])

X_dev_lst = X_dev.values.tolist()
Y_dev_lst = Y_dev.values.tolist()
for x, y in zip(X_dev_lst, Y_dev_lst):
    dataset["dev"].append([x, y[0]])

X_tst_lst = test_X.values.tolist()
Y_tst_lst = [0]*len(X_tst_lst)
for x, y in zip(X_tst_lst, Y_tst_lst):
    dataset["test"].append([x, y])

with open("../data/dataset.json", "w+") as f:
    json.dump(dataset, f)

code.interact(local = locals())
