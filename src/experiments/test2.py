import pandas as pd

train = pd.read_csv("/Users/rshu/Downloads/dl-master/data/total_features/cass/training_set/totalFeatures4.csv")
test = pd.read_csv("/Users/rshu/Downloads/dl-master/data/total_features/cass/test_set/totalFeatures5.csv")

trainHeader = train.columns.tolist()
print(trainHeader)
print(len(trainHeader))

testHeader = test.columns.tolist()
print(testHeader)
print(len(testHeader))

sameItems = []

for i in trainHeader:
    if i in testHeader:
        sameItems.append(i)

# sameItems = list(set(train).intersection(test))
print(sameItems)
print(len(sameItems))
print("F72" in sameItems)

for i in trainHeader:
    if i not in testHeader:
        print("d1: ", i)
        train = train.drop(i, 1)

print(" ")

for i in testHeader:
    if i not in trainHeader:
        print("d2: ", i)
        test = test.drop(i, 1)

print(len(train.columns.tolist()))
print(len(test.columns.tolist()))