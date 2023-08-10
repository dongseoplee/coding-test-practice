#13458번 시험 감독
import sys
import math
n = int(sys.stdin.readline())
numList = list(map(int, sys.stdin.readline().split()))
b, c = map(int, sys.stdin.readline().split())
cntList = []
for i in range(n):
  main = 1
  sub = 0
  if numList[i] - b <= 0:
    cntList.append(main)
  else:
    sub = math.ceil((numList[i] - b) / c)
    cntList.append(main + sub)
  sub = 0

print(sum(cntList))

#14501번 퇴사
import sys
n = int(sys.stdin.readline())
dp = [0] * (n + 1)
schedule = []
for _ in range(n):
  schedule.append(list(map(int, sys.stdin.readline().split())))

for i in range(n-1, -1, -1): #6~0
  if i + schedule[i][0] > n:
    dp[i] = dp[i+1]
  else:
    dp[i] = max(schedule[i][1] + dp[i + schedule[i][0]], dp[i+1])
print(dp[0])


#14888번 연산자 끼워넣기
import sys
from itertools import permutations
n = int(sys.stdin.readline())
nums = list(map(int, sys.stdin.readline().split()))
opNum = list(map(int, sys.stdin.readline().split()))
opList = ['+', '-', '*', '/']
op = []
res = []
for j in range(4):
  for k in range(opNum[j]):
    op.append(opList[j])

# print(op)
maximum = 1e9
minimum = -1e9
def calculator():
  global maximum, minimum
  for case in permutations(op, n-1): #permutation 두번째 인자는 크기를 넣어준다.
    calNum = nums[0]
    for i in range(1, n): #12345
      if case[i-1] == '+':
        calNum += nums[i]
      if case[i-1] == '-':
        calNum -= nums[i]
      if case[i-1] == '*':
        calNum *= nums[i]
      if case[i-1] == '/':
        calNum = int(calNum / nums[i])

    res.append(calNum)

calculator()
# print(res)
print(max(res))
print(min(res))

#14889번 스타트와 링크
import sys
from itertools import combinations
n = int(sys.stdin.readline())
graph = []
member = {i for i in range(n)}
for _ in range(n):
  graph.append(list(map(int, sys.stdin.readline().split())))

# print(member)
# print(graph)
minScore = sys.maxsize
for com in combinations(member, n//2):
  # print(list(com), list(member - set(com)))
  g1 = list(com)
  g2 = list(member - set(com))
  # print(g1, g2)
  g1Score = 0
  g2Score = 0
  for i in g1:
    for j in g1:
      g1Score += graph[i][j]
  for k in g2:
    for l in g2:
      g2Score += graph[k][l]

  minScore = min(minScore, abs(g1Score - g2Score))

print(minScore)


#14503번 로봇 청소기
import sys
n, m = map(int, sys.stdin.readline().split())
r, c, d = map(int, sys.stdin.readline().split())
graph = []
visited = [[False for _ in range(m)] for _ in range(n)]
for _ in range(n):
  graph.append(list(map(int, sys.stdin.readline().split())))

# print(visited)
# print(graph)
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]
while(1):
  nx = r
  ny = c
  if graph[nx][ny] == 0:
    graph[nx][ny] = 'clean'

  if graph[nx+1][ny] != 0 and graph[nx-1][ny] != 0 and graph[nx][ny+1] != 0 and graph[nx][ny-1] != 0:
    if d == 0:
      if graph[nx+1][ny] == 1:
        break
      else:
        r = nx+1
        c = ny
        continue
    if d == 1:
      if graph[nx][ny-1] == 1:
        break
      else:
        r = nx
        c = ny-1
        continue
    if d == 2:
      if graph[nx-1][ny] == 1:
        break
      else:
        r = nx-1
        c = ny
        continue

    if d == 3:
      if graph[nx][ny+1] == 1:
        break
      else:
        r = nx
        c = ny+1
        continue
  if graph[nx+1][ny] == 0 or graph[nx-1][ny] == 0 or graph[nx][ny+1] == 0 or graph[nx][ny-1] == 0:
    d = (d+3)%4
    if d == 0:
      if graph[nx-1][ny] == 0:
        graph[nx-1][ny] = 'clean'
        r = nx - 1
        c = ny
        continue
    elif d == 1:
      if graph[nx][ny+1] == 0:
        graph[nx][ny+1] = 'clean'
        r = nx
        c = ny + 1
        continue
    elif d == 2:
      if graph[nx+1][ny] == 0:
        graph[nx+1][ny] = 'clean'
        r = nx + 1
        c = ny
        continue
    elif d == 3:
      if graph[nx][ny-1] == 0:
        graph[nx][ny-1] = 'clean'
        r = nx
        c = ny - 1
        continue


# print(graph)
cnt = 0
for graphRow in graph:
  for graphRowData in graphRow:
    if graphRowData == 'clean':
      cnt += 1


print(cnt)

#14499번 주사위 굴리기
import sys

n, m, x, y, k = map(int, sys.stdin.readline().split())
graph = []
command = []
dice = [0] * 6
for _ in range(n):
  graph.append(list(map(int, sys.stdin.readline().split())))
command = list(map(int, sys.stdin.readline().split()))
# print(graph)
# print(command)
# print(dice)
for i in range(len(command)):
  a, b, c, d, e, f = dice[0], dice[1], dice[2], dice[3], dice[4], dice[
    5]  # 주사위 상하좌우왼위를 인덱스로 갖고 주사위가 굴러가면 인덱스에 대한 값을 변경한다.
  if command[i] == 1:
    if y + 1 >= m:
      print("", end='')
    else:
      y += 1  # 동으로 굴린다.
      dice[0], dice[1], dice[2], dice[3], dice[4], dice[5] = d, b, a, f, e, c
      if graph[x][y] == 0:
        graph[x][y] = dice[5]
      else:
        dice[5] = graph[x][y]
        graph[x][y] = 0
      print(dice[0])

  if command[i] == 2:
    if y - 1 < 0:
      print("", end='')
    else:
      y -= 1  # 서쪽으로 굴린다.
      dice[0], dice[1], dice[2], dice[3], dice[4], dice[5] = c, b, f, a, e, d
      if graph[x][y] == 0:
        graph[x][y] = dice[5]
      else:
        dice[5] = graph[x][y]
        graph[x][y] = 0
      print(dice[0])

  if command[i] == 3:
    if x - 1 < 0:
      print("", end='')
    else:
      x -= 1  # 북쪽으로 굴린다.
      dice[0], dice[1], dice[2], dice[3], dice[4], dice[5] = e, a, c, d, f, b
      if graph[x][y] == 0:
        graph[x][y] = dice[5]
      else:
        dice[5] = graph[x][y]
        graph[x][y] = 0
      print(dice[0])

  if command[i] == 4:
    if x + 1 >= n:
      print("", end='')
    else:
      x += 1  # 남쪽으로 굴린다.
      dice[0], dice[1], dice[2], dice[3], dice[4], dice[5] = b, f, c, d, a, e
      if graph[x][y] == 0:
        graph[x][y] = dice[5]
      else:
        dice[5] = graph[x][y]
        graph[x][y] = 0
      print(dice[0])


