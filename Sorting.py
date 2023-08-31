#2108번 통계학
import sys
from collections import Counter

n = int(sys.stdin.readline())
arr = []
for _ in range(n):
    arr.append(int(sys.stdin.readline()))

# print(arr)
print(int(round(sum(arr) / n, 0)))
arr.sort()
print(arr[n // 2])
c = Counter(arr)
# print(c.most_common())

if len(c.most_common()) > 1 and (c.most_common()[0][1] == c.most_common()[1][1]):
    print(c.most_common()[1][0])
else:
    print(c.most_common()[0][0])

print(arr[n - 1] - arr[0])

#13164번 행복 유치원
import sys
from itertools import combinations

n, k = map(int, sys.stdin.readline().split())
graph = list(map(int, sys.stdin.readline().split()))
temp = []
for i in range(1, n):
  temp.append(graph[i] - graph[i-1])

# print(temp)
temp.sort()

# 길이의 간격이 높은 수부터 k-1개 제거
print(sum(temp[:n-k]))

#2309번 일곱 난쟁이
import sys
from itertools import combinations

people = [0] * 9
for k in range(9):
  people[k] = int(sys.stdin.readline())

for i in combinations(people, 7): #조합을 for문에 쓸 수 있다.
  # print(i)
  if sum(i) == 100:
    temp = list(i)
    temp.sort()
    for j in range(len(temp)):
      print(temp[j])
    break
