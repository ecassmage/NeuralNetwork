tup = [(1, 2), (2, 3), (3, 2), (2, 1)]
weights = [[1, 2, 3, 4], [1], [2], [4]]
biases = [[1], [2], [3], [4]]

for (a, d), b, c in zip(tup, weights, biases):
    print(a, d, b, c)
