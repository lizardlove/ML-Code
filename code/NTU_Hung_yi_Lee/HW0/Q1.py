import numpy as np

matrixA = []
for i in open("./01-Data/matrixA.txt"):
    row = [int(x) for x in i.split(",")]
    matrixA.append(row)

matrixB = []
for j in open("./01-Data/matrixB.txt"):
    row = [int(x) for x in j.split(",")]
    matrixB.append(row)

matrixA = np.array(matrixA)
matrixB = np.array(matrixB)

ans = matrixA.dot(matrixB)
ans.sort(axis=1)
print matrixA,matrixB
np.savetxt("Q1_ans.txt", ans, fmt="%d", delimiter="\r\n")
