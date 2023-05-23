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

#2579번 계단 오르기
import sys

'''
DP는 이전의 값을 재활용해서 불필요한 반복을 없애고, 그만큼 시간 복잡도를 줄이도록 구현하면 된다.
점화식 생성
동적계획법의 핵심은 첫 몇 개의 케이스를 하드코딩해주고 나머지 경우들을 점화식을 통해 처리
'''
n = int(sys.stdin.readline())
stairList = [0]
dp = [0] * (n + 1)
for _ in range(0, n):
    stairList.append(int(sys.stdin.readline()))

if n == 1:
    print(stairList[1])
elif n == 2:
    print(stairList[1] + stairList[2])
elif n == 3:
    print(max(stairList[1] + stairList[3], stairList[2] + stairList[3]))
else:
    dp[1] = stairList[1]
    dp[2] = dp[1] + stairList[2]
    dp[3] = max(stairList[2] + stairList[3], stairList[1] + stairList[3])
    for i in range(4, n + 1):
        dp[i] = max(dp[i - 2] + stairList[i], dp[i - 3] + stairList[i - 1] + stairList[i])
    print(dp[n])

#1149번 RGB거리
import sys
houseCnt = int(sys.stdin.readline())
houseList = [0] * (houseCnt + 1)
dp = [0] * (houseCnt + 1)
costList = []
for _ in range(houseCnt):
  costList.append(list(map(int, sys.stdin.readline().split())))

# print(costList)

for i in range(1, houseCnt):
  costList[i][0] += min(costList[i-1][1], costList[i-1][2])
  costList[i][1] += min(costList[i-1][0], costList[i-1][2])
  costList[i][2] += min(costList[i-1][1], costList[i-1][0])

print(min(costList[houseCnt-1]))

#11726번 2×n 타일링
import sys
n = int(sys.stdin.readline())
dp = [0] * (n+1)

if n == 1:
  print(1)
elif n == 2:
  print(2)
else:
  dp[1] = 1
  dp[2] = 2
  for i in range(3, n+1):
    dp[i] = dp[i-1] + dp[i-2]
  print(dp[n] % 10007)


#9461번 파도반 수열
import sys

t = int(sys.stdin.readline())
for _ in range(t):
    n = int(sys.stdin.readline())
    dp = [0] * (n + 1)
    if n == 1:
        print(1)
    elif n == 2:
        print(1)
    elif n == 3:
        print(1)
    else:
        dp[1] = 1
        dp[2] = 1
        dp[3] = 1
        for i in range(1, n - 2):  # n = 4 , 1
            dp[i + 3] = dp[i] + dp[i + 1]

        # print(dp)
        print(dp[n])

#1965번 상자넣기
import sys
n = int(sys.stdin.readline())
box = list(map(int, sys.stdin.readline().split()))
dp = [1] * n
for i in range(n):
  for j in range(i):
    if box[i] > box[j]:
      dp[i] = max(dp[i], dp[j] + 1)

print(max(dp))

#11053번 가장 긴 증가하는 부분 수열
import sys
n = int(sys.stdin.readline())
a = list(map(int, sys.stdin.readline().split()))
dp = [1] * n
for i in range(n):
  for j in range(i):
    if a[i] > a[j]:
      dp[i] = max(dp[i], dp[j] + 1)

print(max(dp))

#11055번 가장 큰 증가하는 부분 수열
import sys
n = int(sys.stdin.readline())
a = list(map(int, sys.stdin.readline().split()))
dp = [1] * n
dp[0] = a[0]
for i in range(1, n):
  for j in range(i):
    if a[i] > a[j]:
      dp[i] = max(dp[i], dp[j] + a[i])
    else:
      dp[i] = max(dp[i], a[i])

print(max(dp))
# print(dp)

#1003번 피보나치 함수
import sys

t = int(sys.stdin.readline())
for _ in range(t):
    n = int(sys.stdin.readline())
    if n == 0:
        zeroNum = 1
        oneNum = 0
        print(zeroNum, oneNum)
    elif n == 1:
        zeroNum = 0
        oneNum = 1
        print(zeroNum, oneNum)
    else:
        dp = []
        dp.append([1, 0])
        dp.append([0, 1])
        for i in range(2, n + 1):
            zeroNum = dp[i - 1][0] + dp[i - 2][0]
            oneNum = dp[i - 1][1] + dp[i - 2][1]
            dp.append([zeroNum, oneNum])

        print(dp[n][0], dp[n][1])
