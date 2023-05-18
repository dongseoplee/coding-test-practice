#1463번 1로 만들기
import sys  # 역순으로 접근 n->1 과 1->n 으로 만든는 최소 연산 갯수는 동일하다.

n = int(sys.stdin.readline())
res = [0] * (n + 1)
# print(res)

for i in range(1, n + 1):
    if i == 1:
        continue
    if i % 3 == 0 and i % 2 == 0:
        res[i] = min(res[i // 3] + 1, res[i // 2] + 1, res[i - 1] + 1)
    if i % 3 != 0 and i % 2 == 0:
        res[i] = min(res[i // 2] + 1, res[i - 1] + 1)
    if i % 3 == 0 and i % 2 != 0:
        res[i] = min(res[i // 3] + 1, res[i - 1] + 1)
    if i % 3 != 0 and i % 2 != 0:
        res[i] = res[i - 1] + 1

print(res[n])

#9095번 1, 2, 3 더하기
#1, 2, 3의 합이니까 d[k] = d[k-1] + d[k-2] + d[k-3]
import sys
testNum = int(sys.stdin.readline())
for _ in range(testNum):
  n = int(sys.stdin.readline())
  if n >= 4:
    res = [0] * (n+1)
    res[1] = 1
    res[2] = 2
    res[3] = 4
    for i in range(4, n+1):
      res[i] = res[i-1] + res[i-2] + res[i-3]
    print(res[n])
  if n == 1:
    print(1)
  if n == 2:
    print(2)
  if n == 3:
    print(4)