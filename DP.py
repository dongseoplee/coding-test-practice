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
#2156번 포도주 시식
import sys  # 최대합 구하기 단, 3개 연속은 선택 불가

n = int(sys.stdin.readline())
podo = []
dp = [0] * (n)
for _ in range(n):
    podo.append(int(sys.stdin.readline()))

if n == 1:
    print(podo[0])
elif n == 2:
    print(podo[0] + podo[1])
else:  # 3잔 연속으로 선택 불가
    dp[0] = podo[0]
    dp[1] = podo[0] + podo[1]
    dp[2] = max(podo[0] + podo[2], podo[1] + podo[2], dp[1])  # dp[1]은 podo[0] + podo[1]과 같다.
    for i in range(3,
                   n):  # i번째가 선택안될때 (앞에 연속 2개가 선택된 경우), i가 선택될떄 (i랑 i-1번째가 선택됨) i가 2일때를 예시로 생각해보기 (0, 2) (1, 2) (0, 1)
        dp[i] = max(dp[i - 2] + podo[i], dp[i - 3] + podo[i - 1] + podo[i], dp[i - 1])
    print(dp[n - 1])

#1921번 연속합
import sys
n = int(sys.stdin.readline())
inputList = list(map(int, sys.stdin.readline().split()))

# print(inputList)
for i in range(1, n):
  inputList[i] = max(inputList[i], inputList[i-1] + inputList[i])

print(max(inputList))

#11727번 2×n 타일링 2
import sys
n = int(sys.stdin.readline())
dp = [0] * (n+1)
if n == 1:
  print(1)
elif n == 2:
  print(3)
else:
  dp[1] = 1
  dp[2] = 3
  for i in range(3, n+1):
    dp[i] = dp[i-1] + dp[i-2]*2

  print(dp[n]%10007)

#2193번 이친수
import sys #dp 단계별로 적어보고 그려보면서  앞 앞앞을 보고 규칙을 찾아라
n = int(sys.stdin.readline())
if n == 1:
  print(1)
elif n == 2:
  print(1)
else:
  dp = [0] * (n)
  dp[0] = 1
  dp[1] = 1
  for i in range(2, n):
    dp[i] = dp[i-2] + dp[i-1]
  print(dp[n-1])

#24416번 알고리즘 수업 - 피보나치 수 1
import sys
n = int(sys.stdin.readline())
dpRes = n - 2
#n은 5이상
dp = [0] * (n+1)
dp[1] = 1
dp[2] = 1
for i in range(3, n+1):
  dp[i] = dp[i-2] + dp[i-1]

print(dp[n], end=' ')
print(dpRes)

#10844번 쉬운 계단 수
import sys
n = int(sys.stdin.readline()) #2차원 배열을 생성해서 규칙찾아보기
dp = [[0 for _ in range(10)] for _ in range(n+1)]

for i in range(n+1): #행
  for j in range(10): #열
    if i == 1 and j >= 1:
      dp[i][j] = 1
    elif i == 1 and j == 0:
      dp[i][j] = 0
    elif i != 1 and j == 0:
      dp[i][j] = dp[i-1][j+1]
    elif i != 1 and j == 9:
      dp[i][j] = dp[i-1][j-1]
    else:
      dp[i][j] = dp[i-1][j-1] + dp[i-1][j+1]

print(sum(dp[n])%1000000000)

#1904번 01타일
import sys
n = int(sys.stdin.readline())
if n == 1:
  print(1)
elif n == 2:
  print(2)
else:
  dp = [0] * (n+1)
  dp[1] = 1
  dp[2] = 2
  for i in range(3, n+1):
    dp[i] = (dp[i-1] + dp[i-2])%15746 #합의 나머지를 다시 합해서 나머지와 같다...???
  print(dp[n])

#11052번 카드 구매하기
import sys
n = int(sys.stdin.readline())
price = [0] + list(map(int, sys.stdin.readline().split()))

dp = [0] * (n+1)
for j in range(1, n+1): #dp[]
  for i in range(1, j+1):
    dp[j] = max(dp[j], dp[j-i] + price[i])

print(dp[n])

#9655번 돌 게임
import sys
n = int(sys.stdin.readline())
# dp = [0] * (n+1)
if n % 2 == 0:
  print('CY')
else:
  print('SK')
