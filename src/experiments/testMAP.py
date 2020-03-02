from algorithms import MAP

actual = [0, 0, 0, 0, 0, 0, 0, 0]
predicted = [0, 0, 0, 0, 0, 0, 1, 0]

for k in range(1, 9):
    print(MAP.apk(actual, predicted, k))
