#1978번 소수 찾기

import math

def primeNumber(x):
    if x == 1:
        return False
    for i in range(2, int(math.sqrt(x) + 1)):
        if x % i == 0:
            return False

    return True


cnt = int(input())
inputList = list(map(int, input().split()))

res = []
for i in range(len(inputList)):
    res.append(primeNumber(inputList[i]))

# print(res)

answer = 0
for input in res:
    if input == True:
        answer = answer + 1
print(answer)


#2581번 소수
import math

def primeNumber(x):
    if x == 1:
        return False
    for i in range(2, int(math.sqrt(x) + 1)):
        if x % i == 0:
            return False

    return True


M = int(input())
N = int(input())

answer = []
for i in range(M, N + 1):
    if primeNumber(i):
        answer.append(i)

if len(answer) == 0:
    print(-1)
else:
    print(sum(answer))
    print(min(answer))

#11653 소인수분해
num = int(input())
res = []
i = 2
while i <= num:
  if num % i == 0:
    res.append(i)
    num = num // i
    i = 2
  else:
    i += 1

for i in range(len(res)):
  print(res[i])