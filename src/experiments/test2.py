import pandas as pd

train_df = pd.read_csv("/home/rshu/Documents/e-dom/data/STATICWARNING/ant-train.csv")
test_df = pd.read_csv("/home/rshu/Documents/e-dom/data/STATICWARNING/ant-test.csv")

print(train_df.head())

train_df.loc[train_df['label'] == 'open', 'label'] = 0
train_df.loc[train_df['label'] == 'close', 'label'] = 1
#
# train_df['label'][train_df.label == 'open'] = 0
# train_df['label'][train_df.label == 'close'] = 1

print(train_df.head())




# trainHeader = train.columns.tolist()
# print(trainHeader)
# print(len(trainHeader))
#
# testHeader = test.columns.tolist()
# print(testHeader)
# print(len(testHeader))
#
# sameItems = []
#
# for i in trainHeader:
#     if i in testHeader:
#         sameItems.append(i)
#
# # sameItems = list(set(train).intersection(test))
# print(sameItems)
# print(len(sameItems))
# print("F72" in sameItems)
#
# for i in trainHeader:
#     if i not in testHeader:
#         print("d1: ", i)
#         train = train.drop(i, 1)
#
# print(" ")
#
# for i in testHeader:
#     if i not in trainHeader:
#         print("d2: ", i)
#         test = test.drop(i, 1)
#
# print(len(train.columns.tolist()))
# print(len(test.columns.tolist()))