#11399번 ATM
import sys
n = int(sys.stdin.readline())
greedy = list(map(int, sys.stdin.readline().split()))
greedy = sorted(greedy)
sum = greedy[0]
for i in range(1, n):
  greedy[i] = greedy[i-1] + greedy[i]
  sum += greedy[i]


print(sum)

#11047번 동전 0
import sys
n, k = map(int, sys.stdin.readline().split())
price = [0] * n
cnt = [0] * n
for i in range(n):
  price[i] = int(sys.stdin.readline())

price.sort(reverse=True)

cnt = 0
#4500을 50000으로 나눈 몫은 0이다.
# 나눠지는 몫이 있으면 몫이 갯수고 k는 나머지 값으로 바뀐다.
for j in price:
  if k == 0:
    break
  cnt += k//j
  k %= j
print(cnt)

#1931번 회의실 배정
import sys #회의 끝나는 시간 기준으로 정렬, 회의가 빨리 끝나야 최대한 많은 회의 가능하다.
n = int(sys.stdin.readline())
conference = []
for i in range(n):
  conference.append(list(map(int, sys.stdin.readline().split())))
conference.sort(key=lambda x:(x[1], x[0]))
cnt = 1
endTime = conference[0][1]
for j in range(1, n):
  if conference[j][0] >= endTime:
    cnt+=1
    endTime = conference[j][1]

print(cnt)

#5585번 거스름돈
import sys
n = int(sys.stdin.readline())
price = 1000 - n
coins = [500, 100, 50, 10, 5, 1]
coinsNum = [0] * 6
# print(coins)
# print(coinsNum)
for i in range(6):
  coinsNum[i] = price // coins[i]
  price = price % coins[i]

print(sum(coinsNum))

#1541번 잃어버린 괄호
import sys
exp = (sys.stdin.readline().split('-'))
res = 0
for i in exp[0].split('+'):
  res += int(i)
for i in exp[1:]:
  for j in i.split('+'):
    res -= int(j)
print(res)

#2217번 로프
import sys  # 그리디 모든 경우의 수 다 해본다.

n = int(sys.stdin.readline())

rope = []
greedy = [0] * n
for _ in range(n):
  rope.append(int(sys.stdin.readline()))

# sorted(rope, reverse=True)
rope.sort(reverse=True)  # 내림차순 정렬해서 i+1개 중에서는 i번째가 최소값이다.
# print(rope)
for i in range(n):
  greedy[i] = rope[i] * (i + 1)

print(max(greedy))

#1789번 수들의 합
import sys

n = int(sys.stdin.readline())
plusNum = 0
res = 0
for i in range(1, n + 1):
  plusNum += i
  res += 1
  if n < plusNum:
    res -= 1
    break

print(res)

#13305번 주유소
import sys

# 매번 최소값을 업데이트 시켜서 가격 책정
n = int(sys.stdin.readline())
dis = list(map(int, sys.stdin.readline().split()))
price = list(map(int, sys.stdin.readline().split()))
minPrice = price[0]
sum = 0
for i in range(n - 1):
  minPrice = min(price[i], minPrice)
  sum += dis[i] * minPrice

print(sum)

#10162번 전자레인지
import sys
num = int(sys.stdin.readline())
button = [300, 60, 10] * 3
count = [0] * 3
for i in range(3):
  count[i] = num // button[i]
  num = num % button[i]

if num == 0:
  print(*count)
else:
  print(-1)

#10610번 30
import sys

num = str(sys.stdin.readline().rstrip())
sum = 0
digit = []
if '0' not in num:
  print(-1)
  exit()
else:
  for i in range(len(num)):
    sum += int(num[i])

  if sum % 3 != 0:
    print(-1)
  else:
    sortedNum = sorted(num, reverse=True)  # 내림차순 정렬
    res = "".join(sortedNum)  # 배열에 문자 하나씩 들어있는 sortNum 배열을 문자열로 변환
    print(res)

#4796번 캠핑
import sys
cnt = 0
while(1):
  cnt += 1
  l, p, v = map(int, sys.stdin.readline().split())
  if l == 0 and p == 0 and v == 0:
    break
  #나머지가 l보다 작냐 크냐 구분 필요
  if v%p >= l:
    day = l*(v//p) + l
  else:
    day = l*(v//p) + v%p
  print("Case ", end = '')
  print(cnt, end='')
  print(": ", end='')
  print(day)


#2864번 5와 6의 차이
import sys #문자열 replace 함수를 사용해서 하면 코드가 간결해진다.

num1, num2 = map(int, sys.stdin.readline().split())
tempNum1, tempNum2 = num1, num2
minNum1, minNum2, maxNum1, maxNum2 = 0, 0, 0, 0

cnt = 0
# 최소 5, 6 -> 5
while num1 != 0:
  if num1 % 10 == 5 or num1 % 10 == 6:
    minNum1 = minNum1 + (5) * (10 ** (cnt))
  else:
    minNum1 = minNum1 + (num1 % 10) * (10 ** (cnt))
  num1 = num1 // 10
  cnt += 1

cnt = 0
while num2 != 0:
  if num2 % 10 == 5 or num2 % 10 == 6:
    minNum2 = minNum2 + (5) * (10 ** (cnt))
  else:
    minNum2 = minNum2 + (num2 % 10) * (10 ** (cnt))
  num2 = num2 // 10
  cnt += 1

resMin = minNum1 + minNum2
# print(minNum1, minNum2)
# print(resMin)
num1, num2 = tempNum1, tempNum2

cnt = 0
while num1 != 0:
  if num1 % 10 == 5 or num1 % 10 == 6:
    maxNum1 = maxNum1 + (6) * (10 ** (cnt))
  else:
    maxNum1 = maxNum1 + (num1 % 10) * (10 ** (cnt))
  num1 = num1 // 10
  cnt += 1

cnt = 0
while num2 != 0:
  if num2 % 10 == 5 or num2 % 10 == 6:
    maxNum2 = maxNum2 + (6) * (10 ** (cnt))
  else:
    maxNum2 = maxNum2 + (num2 % 10) * (10 ** (cnt))
  num2 = num2 // 10
  cnt += 1

resMax = maxNum1 + maxNum2
print(resMin, resMax)
# 최대 5, 6 -> 6

# print(num1)
# print(num2)

#1715번 카드 정렬하기
import sys  # 우선 순위  heapq.heappush, heapq.heappop
import heapq

n = int(sys.stdin.readline())
cards = []
for _ in range(n):
  heapq.heappush(cards, int(sys.stdin.readline()))

res = 0
if len(cards) == 1:
  print(res)
else:
  for _ in range(n - 1):
    num1 = heapq.heappop(cards)
    num2 = heapq.heappop(cards)
    res += num1 + num2
    heapq.heappush(cards, num1 + num2)
  print(res)

#16953번 A → B
import sys
from collections import deque

a, b = map(int, sys.stdin.readline().split())
queue = deque()
queue.append((a, 1))  # 숫자와 횟수를 넣는다. (숫자, 횟수)


def bfs():
  while queue:

    i, cnt = queue.popleft()

    if i * 2 == b or (i * 10) + 1 == b:
      print(cnt + 1)
      exit()
    if i * 2 <= b:  # 큐에 넣을 수 있는 제한 조건을 b의 값과 크기 비교로 한다.
      queue.append((i * 2, cnt + 1))
    if (i * 10) + 1 <= b:
      queue.append(((i * 10) + 1, cnt + 1))
  print(-1)


bfs()

#1439번 뒤집기
import sys

s = list(sys.stdin.readline().rstrip())
cnt = 0
for i in range(len(s) - 1):
  if s[i] != s[i+1]:
    cnt += 1

print((cnt + 1) // 2)

#1543번 문서 검색
import sys
str1 = sys.stdin.readline().rstrip()
str2 = sys.stdin.readline().rstrip()
idx = 0
cnt = 0
while idx < (len(str1)-len(str2)+1):
  temp = str1[idx:idx+len(str2)]
  if temp == str2:
    cnt += 1
    idx += len(str2)
    # print(idx)
  else:
    idx += 1

print(cnt)

#2578번 빙고
import sys

graph = []
num = []
visited = [[False for _ in range(5)] for _ in range(5)]
for _ in range(5):
  graph.append(list(map(int, sys.stdin.readline().split())))
for _ in range(5):
  num.append(list(map(int, sys.stdin.readline().split())))


# print(graph, num)
# a = [1, 2, 3, 3]
# print(a.count(3))
def bingGo(temp):
  cnt = 0
  for i in range(5):  # 가로
    if temp[i].count(True) == 5:
      # print("가로")
      cnt += 1
  for j in range(5):
    col = []
    for k in range(5):
      col.append(temp[k][j])
    if col.count(True) == 5:
      # print("세로")

      cnt += 1

  col2 = []
  for l in range(5):  # 좌상향 대각
    col2.append(temp[l][l])
  if col2.count(True) == 5:
    # print("우대각")
    cnt += 1

  col3 = []
  for m in range(5):  # 우상향 대각
    col3.append(temp[4 - m][m])
  if col3.count(True) == 5:
    # print("좌대각")

    cnt += 1

  return cnt


# bbb = [[False, False, True, False, True], [True, True, True, True, True], [True, False, True, False, True], [False, True, True, False, False], [True, False, True, False, True]]
# print(bingGo(bbb))
for a in range(25):
  tellNum = num[a // 5][a % 5]
  for b in range(5):
    for c in range(5):
      if graph[b][c] == tellNum:
        visited[b][c] = True
      if bingGo(visited) >= 3:
        print(a + 1)
        # print(visited)
        exit()

