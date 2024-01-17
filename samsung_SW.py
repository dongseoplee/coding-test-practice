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


#14500번 테트로미노
import sys

n, m = map(int, sys.stdin.readline().split())
graph = []
for _ in range(n):
  graph.append(list(map(int, sys.stdin.readline().split())))

# print(graph)
res = []


def rec1(i, j):  # 3x2
  temp = []
  for k in range(i, i + 3):
    temp.append(graph[k][j:j + 2])
  # print("temp", temp)
  res.append(temp[0][0] + temp[1][0] + temp[1][1] + temp[2][0])  # 1
  res.append(temp[0][1] + temp[1][0] + temp[1][1] + temp[2][1])  # 2

  res.append(temp[0][0] + temp[1][0] + temp[1][1] + temp[2][1])  # 3
  res.append(temp[0][1] + temp[1][0] + temp[1][1] + temp[2][0])  # 4

  res.append(temp[0][0] + temp[1][0] + temp[2][0] + temp[2][1])  # 5
  res.append(temp[0][1] + temp[1][1] + temp[2][0] + temp[2][1])  # 6

  res.append(temp[0][0] + temp[0][1] + temp[1][1] + temp[2][1])  # 7
  res.append(temp[0][0] + temp[0][1] + temp[1][0] + temp[2][0])  # 8


def rec2(i, j):  # 2x3
  temp = []
  for k in range(i, i + 2):
    temp.append(graph[k][j:j + 3])
  # print("temp", temp)
  res.append(temp[0][1] + sum(temp[1]))  # 1
  res.append(sum(temp[0]) + temp[1][1])  # 2

  res.append(temp[0][1] + temp[0][2] + temp[1][0] + temp[1][1])  # 3
  res.append(temp[0][0] + temp[0][1] + temp[1][1] + temp[1][2])  # 4

  res.append(temp[0][2] + temp[1][0] + temp[1][1] + temp[1][2])  # 5
  res.append(temp[0][0] + temp[0][1] + temp[0][2] + temp[1][2])  # 6

  res.append(temp[0][0] + temp[0][1] + temp[0][2] + temp[1][0])  # 7
  res.append(temp[0][0] + temp[1][0] + temp[1][1] + temp[1][2])  # 8


def rec3(i, j):  # 1x4
  temp = []
  temp.append(graph[i][j:j + 4])
  # print("temp", temp)
  res.append(temp[0][0] + temp[0][1] + temp[0][2] + temp[0][3])  # 1


def rec4(i, j):  # 4x1
  temp = [graph[i][j], graph[i + 1][j], graph[i + 2][j], graph[i + 3][j]]
  # print("temp", temp)
  res.append(sum(temp))  # 1


def rec5(i, j):  # 2x2
  temp = []
  for k in range(i, i + 2):
    temp.append(graph[k][j:j + 2])
  # print(temp)

  res.append(temp[0][0] + temp[0][1] + temp[1][0] + temp[1][1])


for i in range(n - 2):  # 0123
  for j in range(m - 1):  # 012
    rec1(i, j)
for i2 in range(n - 1):
  for j2 in range(m - 2):
    rec2(i2, j2)
for i3 in range(n):
  for j3 in range(m - 3):
    rec3(i3, j3)
for i4 in range(n - 3):
  for j4 in range(m):
    rec4(i4, j4)
for i5 in range(n - 1):
  for j5 in range(m - 1):
    rec5(i5, j5)

print(max(res))

#3190번 뱀
import sys
from collections import deque

n = int(sys.stdin.readline())
graph = [[0 for _ in range(n)] for _ in range(n)]
k = int(sys.stdin.readline())
for _ in range(k):
  a, b = map(int, sys.stdin.readline().split())
  graph[a - 1][b - 1] = 1

l = int(sys.stdin.readline())
direction = []
for _ in range(l):
  a, b = sys.stdin.readline().split()
  direction.append((int(a), b))

dx = [0, 1, 0, -1]
dy = [1, 0, -1, 0]
time = 0
idx = 0
i = 0
nx, ny = 0, 0
queue = deque()
queue.append((nx, ny))
while (1):
  nx = nx + dx[idx]
  ny = ny + dy[idx]
  time += 1

  # print(queue)
  # print()
  if nx < 0 or nx >= n or ny < 0 or ny >= n or (nx, ny) in queue:
    # print('nx, ny', nx, ny)
    break
  queue.append((nx, ny))
  if graph[nx][ny] == 0:
    queue.popleft()
  else:
    graph[nx][ny] = 0

  if time == direction[i][0]:  # 방향전환 8가지 경우로 나눌필요 없음
    if direction[i][1] == 'L':
      idx = (idx + 3) % 4
    else:
      idx = (idx + 1) % 4

    if i + 1 < len(direction):
      i += 1

print(time)
# print(queue)
# print(time)

#1249번 보급로
from collections import deque

# 이동거리에 상관없이 최소 값으로 이동하기
testNum = int(input())
for testnum in range(testNum):
  n = int(input())
  graph = []
  visited = [[10e6 for _ in range(n)] for _ in range(n)]
  for _ in range(n):
    graph.append(input().rstrip())

  # print(graph)
  queue = deque()
  dx = [-1, 1, 0, 0]
  dy = [0, 0, -1, 1]
  queue.append((0, 0))
  visited[0][0] = 0
  while queue:
    x, y = queue.popleft()
    for i in range(4):
      nx = x + dx[i]
      ny = y + dy[i]
      if nx < 0 or nx >= n or ny < 0 or ny >= n:
        continue
      else:
        if visited[x][y] + int(graph[nx][ny]) < visited[nx][ny]: #방문여부 상관없음
          visited[nx][ny] = visited[x][y] + int(graph[nx][ny])
          queue.append((nx, ny))

  print('#{}'.format(testnum + 1), end=' ')
  print(visited[n - 1][n - 1])

#1226번 미로1
from collections import deque

for _ in range(10):
  testNum = int(input())
  visited = [[False for _ in range(16)] for _ in range(16)]
  graph = [list(map(int, list(input()))) for _ in range(16)]  # 스페이스 없이 주어지는 값 입력 받는 방법 ex)1111111111111111

  sx, sy = 0, 0
  fx, fy = 0, 0
  for i in range(16):
    for j in range(16):
      if graph[i][j] == 2:
        sx, sy = i, j
      elif graph[i][j] == 3:
        fx, fy = i, j

  # print(sx, sy)
  # print(fx, fy)
  queue = deque()
  # print(graph)
  dx = [-1, 1, 0, 0]
  dy = [0, 0, -1, 1]

  queue.append((sx, sy))
  visited[sx][sy] = True
  while queue:
    x, y = queue.popleft()
    for k in range(4):
      nx = x + dx[k]
      ny = y + dy[k]
      if nx < 0 or nx >= 16 or ny < 0 or ny >= 16:
        continue
      else:
        if (graph[nx][ny] == 0 or graph[nx][ny] == 3) and visited[nx][ny] == False:
          queue.append((nx, ny))
          visited[nx][ny] = True

  flag = 0
  if visited[fx][fy] == True:
    flag = 1

  print('#{}'.format(testNum), end=' ')
  print(flag)


#14890번 경사로
import sys

n, l = map(int, sys.stdin.readline().split())
graph = []
for _ in range(n):
  graph.append(list(map(int, sys.stdin.readline().split())))

# print(graph)
cnt = []


def isRoad(road):
  for j in range(1, n):
    if abs(road[j - 1] - road[j]) > 1:
      return False

    if road[j] < road[j - 1]:  # 오른쪽이 더 낮은 경우 (내리막 경사로)
      for k in range(l):  # l:2 k:0,1
        if j + k >= n or used[j + k] == True or road[j] != road[j + k]:  # 시작점과 높이가 계속 동일하면 된다.
          return False
        if road[j] == road[j + k]:
          used[j + k] = True

    elif road[j] > road[j - 1]:  # 왼쪽이 더 낮은 경우 (오르막 경사로)
      for k in range(l):
        if j - k - 1 < 0 or used[j - k - 1] == True or road[j - 1] != road[j - k - 1]:
          return False
        if road[j - 1] == road[j - k - 1]:
          used[j - k - 1] = True

  return True


# print(isRoad(graph[0]))
# 가로 길
for i in range(n):
  used = [False for _ in range(n)]
  if isRoad(graph[i]) == True:
    cnt.append(True)

# 세로 길
for j in range(n):
  tempList = []
  used = [False for _ in range(n)]
  for k in range(n):
    tempList.append(graph[k][j])

  if isRoad(tempList) == True:
    cnt.append(True)

print(len(cnt))

#14891번 톱니바퀴
import sys
from collections import deque

#재귀함수로 왼쪽 오른쪽, 왼쪽 톱니들을 확인한다.
def check_right(idx, d):
  if idx > 4:
    return

  if q[idx-1][2] != q[idx][6]:
    check_right(idx + 1, d*-1)
    q[idx].rotate(d)

def check_left(idx, d):
  if idx < 1:
    return
  if q[idx+1][6] != q[idx][2]:
    check_left(idx - 1, d*-1)
    q[idx].rotate(d)

q = {}
for i in range(1, 5):
  q[i] = deque(list(map(int, list(sys.stdin.readline().rstrip()))))

k = int(sys.stdin.readline())
for _ in range(k):
  num, direction = map(int, sys.stdin.readline().split())

  check_right(num + 1, direction*-1)
  check_left(num - 1, direction*-1)
  q[num].rotate(direction)

res = 0
for i in range(4):
  res += (2**i)*q[i+1][0]

print(res)

#15683번 감시
import sys
# 조합별로 사각지대를 판단, 모드가 여러가지가 있다면 한 리스트에 모든 cctv를 구현
import copy

n, m = map(int, sys.stdin.readline().split())
graph = []
cctv = []
mode = [
  [],
  [[0], [1], [2], [3]],
  [[0, 2], [1, 3]],
  [[0, 1], [1, 2], [2, 3], [0, 3]],
  [[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]],
  [[0, 1, 2, 3]],
]
dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]

for i in range(n):
  data = list(map(int, sys.stdin.readline().split()))
  graph.append(data)
  for j in range(m):
    if data[j] in [1, 2, 3, 4, 5]:
      cctv.append([data[j], i, j])


# print(graph)

def fill(board, mode, x, y):  # 감시
  for i in mode:
    nx = x
    ny = y
    while True:
      nx += dx[i]
      ny += dy[i]
      if nx < 0 or nx >= n or ny < 0 or ny >= m:  # nx는 n, ny는 m 다른다!!
        break
      if board[nx][ny] == 6:
        break
      elif board[nx][ny] == 0:
        board[nx][ny] = 7


def dfs(depth, graph):
  global minNum
  if depth == len(cctv):
    cnt = 0  # 사각지대 갯수
    for i in range(n):
      cnt += graph[i].count(0)  # 리스트에서 0인값 갯수 세기
    minNum = min(minNum, cnt)
    return

  temp = copy.deepcopy(graph)
  cctvNum, x, y = cctv[depth]
  for i in mode[cctvNum]:
    fill(temp, i, x, y)
    dfs(depth + 1, temp)
    temp = copy.deepcopy(graph)


minNum = int(1e9)
dfs(0, graph)
print(minNum)

#15686번 치킨 배달
import sys
from itertools import combinations

n, m = map(int, sys.stdin.readline().split())

graph = []
houseNum = []
storeNum = []
for _ in range(n):
  graph.append(list(map(int, sys.stdin.readline().split())))

for i in range(n):
  for j in range(n):
    if graph[i][j] == 1:
      houseNum.append((i, j))
    elif graph[i][j] == 2:
      storeNum.append((i, j))

dist = [[0] * len(storeNum) for _ in range(len(houseNum))]
for k in range(len(houseNum)):  # 0123
  for l in range(len(storeNum)):  # 012
    # k번 집에서 l번 치킨집으로 가는 거리 -> dist[k][l]
    hx, hy = houseNum[k]
    sx, sy = storeNum[l]
    dist[k][l] = abs(hx - sx) + abs(hy - sy)

# print(dist)
storeList = [i for i in range(len(storeNum))]

resFinal = []
res = []
ans = []
for st in combinations(storeList, m):
  ans = []
  for ii in range(len(houseNum)):
    distance = sys.maxsize
    res = []
    for jj in range(len(st)):
      res.append(dist[ii][st[jj]])  # 0번 집과 선택된 m개 치킨집까지의 거리 총 m개가 res에 담긴다.
    ans.append(min(res))

  resFinal.append(sum(ans))

print(min(resFinal))

#15685번 드래곤 커브
import sys
import copy

n = int(sys.stdin.readline())
graph = [[False for _ in range(101)] for _ in range(101)]
dx = [0, -1, 0, 1]
dy = [1, 0, -1, 0]

for i in range(n):
  y, x, d, g =  map(int, sys.stdin.readline().split()) # x로 로 준 값을 나는 y값으로 보겠다. y=x 대칭
  graph[x][y] = True

  direction = [d]
  for j in range(g):
    for k in range(len(direction) - 1, -1, -1):
      direction.append((direction[k] + 1)% 4)

  for l in range(len(direction)):
    x = x + dx[direction[l]]
    y = y + dy[direction[l]]
    if x < 0 or x >= 101 or y < 0 or y >= 101:
      continue
    else:
      graph[x][y] = True

res = 0
for m in range(100):
  for n in range(100):
    if graph[m][n] == True and graph[m+1][n] == True and graph[m][n+1] == True and graph[m+1][n+1] == True:
      res += 1

print(res)

#17144번 미세먼지 안녕!
import sys
r, c, t = map(int, sys.stdin.readline().split())
room = [list(map(int, sys.stdin.readline().split())) for _ in range(r)]
front = 0
back = 0

# 공기 청정기 위치 확인
for i in range(r) :
    if room[i][0] == -1 :
        front = i
        back = i + 1
        break

# 미세먼지 확산
def spread() :
    # 상 하 좌 우
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]

    temp = [[0] * c for _ in range(r)]
    for i in range(r) :
        for j in range(c) :
            if room[i][j] != 0 and room[i][j] != -1 :
                value = 0
                for k in range(4) :
                    nx = i + dx[k]
                    ny = j + dy[k]
                    if 0 <= nx < r and 0 <= ny < c and room[nx][ny] != -1 :
                        temp[nx][ny] += room[i][j] // 5
                        value += room[i][j] // 5
                room[i][j] -= value
    for i in range(r) :
        for j in range(c) :
            room[i][j] += temp[i][j]

# 위쪽 공기청정기 동작
def air_up() :
    # 반시계 방향 (동 북 서 남)
    dx = [0, -1, 0, 1]
    dy = [1, 0, -1, 0]
    direct = 0
    before = 0
    x, y = front, 1
    while True :
        nx = x + dx[direct]
        ny = y + dy[direct]
        if x == front and y == 0 :
            break
        if not 0 <= nx < r or not 0 <= ny < c :
            direct += 1
            continue
        room[x][y], before = before, room[x][y]
        x, y = nx, ny

# 아래쪽 공기청정기 동작
def air_down() :
    # 시계 방향 (동 남 서 북)
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    direct = 0
    before = 0
    x, y = back, 1
    while True :
        nx = x + dx[direct]
        ny = y + dy[direct]
        if x == back and y == 0 :
            break
        if not 0 <= nx < r or not 0 <= ny < c :
            direct += 1
            continue
        room[x][y], before = before, room[x][y]
        x, y = nx, ny

for _ in range(t) :
    spread()
    air_up()
    air_down()

result = 0
for i in range(r) :
    for j in range(c) :
        if room[i][j] > 0 :
            result += room[i][j]

print(result)

#16235번 나무 재테크
import sys
from collections import deque

n, m, k = map(int, sys.stdin.readline().split())
a = []
for _ in range(n):
  a.append(list(map(int, sys.stdin.readline().split())))
land = [[5] * n for _ in range(n)]
trees = [[deque() for _ in range(n)] for _ in range(n)] #deque으로 설정하는 것

for _ in range(m):
  x, y, z = map(int, sys.stdin.readline().split())
  trees[x-1][y-1].append(z)
# print(trees)
def spring_summer():
  for i in range(n):
    for j in range(n):
      for k in range(len(trees[i][j])):
        if trees[i][j][k] <= land[i][j]:
          land[i][j] -= trees[i][j][k]
          trees[i][j][k] += 1
        else:
          for _ in range(k, len(trees[i][j])):
            land[i][j] += trees[i][j].pop()//2
          break
dx = [-1, -1, -1, 0, 0, 1, 1, 1]
dy = [-1, 0, 1, -1, 1, -1, 0, 1]
def fall_winter():
  for x in range(n):
    for y in range(n):
      for k in range(len(trees[x][y])):
        if trees[x][y][k] % 5 == 0:
          for d in range(8):
            nx = x + dx[d]
            ny = y + dy[d]
            if 0 <= nx < n and 0 <= ny < n:
              trees[nx][ny].appendleft(1)
      land[x][y] += a[x][y]
# def fall_winter():
#     for x in range(n):
#         for y in range(n):
#             for k in range(len(trees[x][y])):
#                 if trees[x][y][k] % 5 == 0:
#                     for d in range(8):
#                         nx, ny, = x + dx[d], y + dy[d]
#                         if 0 <= nx < n and 0 <= ny < n:
#                             trees[nx][ny].appendleft(1)
#             land[x][y] += a[x][y]

for _ in range(k):
  spring_summer()
  fall_winter()

res = 0
for i in range(n):
  for j in range(n):
    res += len(trees[i][j])

print(res)

#21610번 마법사 상어와 비바라기
n, m = map(int, input().split())  # n*n, m번 이동
graph = []
commands = []

for _ in range(n):
  temp = list(map(int, input().split()))
  graph.append(temp)

for _ in range(m):
  a, b = map(int, input().split())
  commands.append((a, b))

dx = [0, 0, -1, -1, -1, 0, 1, 1, 1]
dy = [0, -1, -1, 0, 1, 1, 1, 0, -1]
dx4 = [-1, -1, 1, 1]
dy4 = [-1, 1, -1, 1]
clouds = [[n - 1, 0], [n - 1, 1], [n - 2, 0], [n - 2, 1]]

for d, s in commands:  # 명령어 횟수
  for i in range(len(clouds)):  # 구름 이동
    nx = (clouds[i][0] + dx[d] * s) % n
    ny = (clouds[i][1] + dy[d] * s) % n
    clouds[i][0] = nx
    clouds[i][1] = ny
  # 구름 이동 완료
  # print("clouds", clouds)
  # print()
  # 비내리기
  for p in range(len(clouds)):
    graph[clouds[p][0]][clouds[p][1]] += 1
  #
  for x, y in clouds:  # 물이 증가한 칸에 물복사 버그 시전 4,2 파악
    waterCnt = 0  # 대각선에 물이 있는 바구니 수
    for j in range(4):
      nx = x + dx4[j]
      ny = y + dy4[j]
      if nx < 0 or nx >= n or ny < 0 or ny >= n:
        continue
      else:
        if graph[nx][ny] > 0:  # 바구니에 물이 있으면
          waterCnt += 1

    graph[x][y] += waterCnt
  # print(graph)
  # print()
  # #구름 삭제
  visited = [[False for _ in range(n)] for _ in range(n)]
  for q in range(len(clouds)):
    visited[clouds[q][0]][clouds[q][1]] = True  # 구름
  clouds = []
  for k in range(n):
    for l in range(n):
      if visited[k][l] == True:
        continue
      else:
        if graph[k][l] >= 2:
          clouds.append([k, l])
          graph[k][l] -= 2
#   print(clouds)
#   print()

#   print(graph)
#   print()

#   #
#   print()

# print(graph)
res = 0
for graphData in graph:
  res += sum(graphData)

print(res)

#21608번 상어 초등학교
n = int(input())
data = [[0] * n for _ in range(n)]
students = [list(map(int, input().split())) for _ in range(n ** 2)]

dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

for student in students:
  available = []

  for i in range(n):
    for j in range(n):
      # 빈자리가 있다면
      if data[i][j] == 0:
        prefer, empty = 0, 0

        # 동서남북 방향 확인하여
        for k in range(4):
          nx = i + dx[k]
          ny = j + dy[k]

          # 범위내에 있을 때
          if 0 <= nx < n and 0 <= ny < n:
            # 좋아하는 학생이 주위에 있다면 더해준다.
            if data[nx][ny] in student[1:]:
              prefer += 1

            # 빈자리가 있다면 더해준다.
            if data[nx][ny] == 0:
              empty += 1

        available.append((i, j, prefer, empty))
  # 정렬
  available.sort(key=lambda x: (-x[2], -x[3], x[0], x[1]))
  data[available[0][0]][available[0][1]] = student[0]

answer = 0
score = [0, 1, 10, 100, 1000]
students.sort()

for i in range(n):
  for j in range(n):
    count = 0

    for k in range(4):
      nx = i + dx[k]
      ny = j + dy[k]

      if 0 <= nx < n and 0 <= ny < n:
        if data[nx][ny] in students[data[i][j] - 1]:
          count += 1

    answer += score[count]

print(answer)

#23288번 주사위 굴리기2
from collections import deque

n, m, k = map(int, input().split())
graph = []
queue = deque()
nowDirection = 'E'
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]
dice = [1, 3, 5, 4, 2, 6]
for _ in range(n):
  graph.append(list(map(int, input().split())))


# print(graph)

def bfs(i, j):
  visited = [[False for _ in range(m)] for _ in range(n)]
  num = graph[i][j]
  cnt = 0
  queue.append((i, j))
  visited[i][j] = True
  while queue:
    x, y = queue.popleft()
    for i in range(4):
      nx = x + dx[i]
      ny = y + dy[i]
      if nx < 0 or nx >= n or ny < 0 or ny >= m:
        continue
      else:
        if graph[nx][ny] == num and visited[nx][ny] == False:
          cnt += 1
          queue.append((nx, ny))
          visited[nx][ny] = True

  return cnt + 1


# print(bfs(2, 0))

def rotateDice(temp, d):
  if d == 'E':
    arr = [temp[0], temp[1], temp[3], temp[5]]
    temp[0] = arr[2]
    temp[1] = arr[0]
    temp[3] = arr[3]
    temp[5] = arr[1]
    return
    # return temp
  elif d == 'W':
    arr = [temp[0], temp[1], temp[3], temp[5]]
    temp[0] = arr[1]
    temp[1] = arr[3]
    temp[3] = arr[0]
    temp[5] = arr[2]
    return
    # return temp
  elif d == 'S':
    arr = [temp[0], temp[2], temp[4], temp[5]]
    temp[0] = arr[2]
    temp[2] = arr[0]
    temp[4] = arr[3]
    temp[5] = arr[1]
    # return temp
    return

  elif d == 'N':
    arr = [temp[0], temp[2], temp[4], temp[5]]
    temp[0] = arr[1]
    temp[2] = arr[3]
    temp[4] = arr[0]
    temp[5] = arr[2]
    # return temp
    return


def moveDice(d, nx, ny):  # d로 어디로 굴려야하는지 알려줌, nx,ny 굴리기 전 주사위 좌표
  # 리턴 값: 어느 좌표로 이동했고 바닥 수는 무엇인지, 진행 방향은 어디쪽인지
  if d == 'E':  # 동
    if ny + 1 >= m:  # 밖으로 넘어가는 경우
      # 반대방향으로 굴린다. 서
      rotateDice(dice, 'W')
      return 'W', nx, ny - 1
    else:  # 동으로 굴린다.
      rotateDice(dice, 'E')
      return 'E', nx, ny + 1
  elif d == 'W':  # 서
    if ny - 1 < 0:  # 밖으로 넘어감
      # 동으로 굴린다.
      rotateDice(dice, 'E')
      return 'E', nx, ny + 1
    else:  # 서쪽으로 굴린다.
      rotateDice(dice, 'W')
      return 'W', nx, ny - 1
  elif d == 'N':  # 북
    if nx - 1 < 0:
      rotateDice(dice, 'S')
      return 'S', nx + 1, ny
    else:
      rotateDice(dice, 'N')
      return 'N', nx + 1, ny
  elif d == 'S':  # 남
    if nx + 1 >= n:
      rotateDice(dice, 'N')
      return 'N', nx + 1, ny
    else:
      rotateDice(dice, 'S')
      return 'S', nx + 1, ny


# print('dice1', dice)
nowDirection = 'E'
sx = 0
sy = 0
score = 0
for j in range(k):
  nowDirection, sx, sy = moveDice(nowDirection, sx, sy)  # 굴러가는 방향이랑
  # print('dice2', dice)
  score += graph[sx][sy] * bfs(sx, sy)  # 점수 합산

  if dice[5] > graph[sx][sy]:  # 90도 시계
    if nowDirection == 'N':
      nowDirection = 'E'
    elif nowDirection == 'E':
      nowDirection = 'S'
    elif nowDirection == 'W':
      nowDirection = 'N'
    elif nowDirection == 'S':
      nowDirection = 'W'
  elif dice[5] < graph[sx][sy]:
    if nowDirection == 'N':
      nowDirection = 'W'
    elif nowDirection == 'E':
      nowDirection = 'N'
    elif nowDirection == 'W':
      nowDirection = 'S'
    elif nowDirection == 'S':
      nowDirection = 'E'
  # print("asd", nowDirection, sx, sy)

  # print()

print(score)

#20057번 마법사 상어와 토네이도
import sys
n = int(sys.stdin.readline())
graph = []
for _ in range(n):
    graph.append(list(map(int, sys.stdin.readline().split())))
dx = [0, 1, 0, -1] #왼, 아, 오, 위
dy = [-1, 0, 1, 0]

nowX = n//2
nowY = n//2

cnt = 1
idx = 0
cnt2 = 0
checkNum = 2

left = [(-1, 1, 0.01), (1, 1, 0.01), (-1, 0, 0.07), (1, 0, 0.07), (-1, -1, 0.1), (1, -1, 0.1), (-2, 0, 0.02), (2, 0, 0.02), (0, -2, 0.05), (0, -1, 0)]
right = [(x, -y, z) for x, y, z in left]
down = [(-y, x, z) for x, y, z in left]
up = [(y, x, z) for x, y, z in left]

trans = {0: left, 1: down, 2: right, 3: up}


res = 0
def move(x, y, direction):
    global res
    #x, y 의 모래가 이동한다.
    if y < 0:
        return
    total = 0
    for dx, dy, z in direction:
        nx = x + dx
        ny = y + dy
        if z == 0:
            new_sand = graph[x][y] - total
        else:
            new_sand = int(graph[x][y]*z)
            total += new_sand
        if nx < 0 or nx >= n or ny < 0 or ny >= n:
            res += new_sand
        else:
            graph[nx][ny] += new_sand
    graph[x][y] = 0


for _ in range(n*2 - 1):
    for _ in range(cnt):
        nowX += dx[idx]
        nowY += dy[idx]
        move(nowX, nowY, trans[idx])
        # print(nowX, nowY)

        cnt2 += 1
    idx = (idx + 1) % 4
    if cnt2 == checkNum:
        cnt += 1
        checkNum += 2
        cnt2 = 0


print(res)


#17140번 이차원 배열과 연산
import sys
r, c, k = map(int, sys.stdin.readline().split())
graph = []
for _ in range(3):
    graph.append(list(map(int, sys.stdin.readline().split())))

# print(len(graph[0]))
time = 0
#R? C?


#등장횟수 정렬
#R 연산

def calculate(matrix, calType):
    sorted_matrix = []
    max_count = 0
    for i in range(len(matrix)):
        temp = []
        dic = dict()
        for j in range(len(matrix[i])):
            if matrix[i][j] != 0:
                if matrix[i][j] not in dic:
                    dic[matrix[i][j]] = 1
                else:
                    dic[matrix[i][j]] += 1

        for key, value in dic.items():
            temp.append([key, value])
        temp.sort(key=lambda x: [x[1], x[0]])
        C = sum(temp, [])
        max_count = max(max_count, len(C))
        sorted_matrix.append(C)
    for m in sorted_matrix:
        m += [0] * (max_count-len(m))
        if len(m) > 100:
            m = m[:100]

    if calType == 'C':
        return list(zip(*sorted_matrix))
    else:
        return sorted_matrix

res = 0

while True:
    if res > 100:
        res = -1
        break
    if r-1 < len(graph) and c-1 < len(graph[0]):
        if graph[r-1][c-1] == k:
            break
    if len(graph) >= len(graph[0]):  # 행 >= 열
        graph = calculate(graph, "R")
    else:
        graph = calculate(list(zip(*graph)), "C")
    res += 1

print(res)

#17779번 게리맨더링2
import sys
n = int(sys.stdin.readline())
graph = [[]]
for _ in range(n):
    graph.append([0] + list(map(int, sys.stdin.readline().split())))

res = 1e9
# print(graph)
section = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
# print(section)
def calculate(x, y, d1, d2):
    num = [0] * (5)
    section = [[0 for _ in range(n+1)] for _ in range(n+1)]
    #경계선구역
    for i in range(d1 + 1):
        section[x+i][y-i] = 5 #1번 조건
        section[x+d2+i][y+d2-i] = 5 # 4번 조건
    for i in range(d2+1):
        section[x+i][y+i] = 5 # 2번 조건
        section[x+d1+i][y-d1+i] = 5 # 3번 조건
    for i in range(x+1, x+d1+d2):
        flag = False
        for j in range(1, n+1):
            if section[i][j] == 5:
                flag = not flag
            if flag:
                section[i][j] = 5

    for r in range(1, n+1):
        for c in range(1, n+1):
            if 1 <= r < x + d1 and 1 <= c <= y and section[r][c] == 0:
                num[0] += graph[r][c]
            elif 1 <= r <= x + d2 and y < c <= n and section[r][c] == 0:
                num[1] += graph[r][c]

            elif x+d1 <= r <= n and 1 <= c < y-d1+d2 and section[r][c] == 0:
                num[2] += graph[r][c]

            elif x+d2 < r <= n and y-d1+d2 <= c <= n and section[r][c] == 0:
                num[3] += graph[r][c]
            elif section[r][c] == 5:
                num[4] += graph[r][c] # 경계선 안에 구역
    return max(num) - min(num)







# x, y, d1, d2를 정하자
for x in range(1, n+1):
    for y in range(1, n+1):
        for d1 in range(1, n+1):
            for d2 in range(1, n+1):
                if 1 <= x < x + d1 + d2 <= n and 1 <= y - d1 < y < y + d2 <= n:
                    res = min(res, calculate(x, y, d1, d2))

print(res)


#20055번
import sys
from collections import deque #rotate 알아감
n, k = map(int, sys.stdin.readline().split())
belt = deque(map(int, sys.stdin.readline().split()))
robot = deque([0 for _ in range(2*n)])

res = 0
while True:
    belt.rotate(1)
    robot.rotate(1)
    robot[n-1] = 0
    for i in range(n-2, -1, -1):
        if robot[i] == 1 and robot[i+1] == 0 and belt[i+1] != 0:
            robot[i] = 0
            robot[i+1] = 1
            belt[i+1] -= 1
    robot[n-1] = 0
    if belt[0] != 0:
        robot[0] = 1
        belt[0] -= 1
    res += 1
    if belt.count(0) >= k:
        print(res)
        break

#17822번 원판 돌리기
from collections import deque
n, m, t = map(int, input().split())
data = [deque(int(x) for x in input().split()) for _ in range(n)]

for tc in range(t) :
    x, d, k = map(int, input().split())
    # 회전 작업 수행
    result = 0
    for i in range(n) :
        result += sum(data[i])
        if (i+1) % x == 0 :
            if d == 0 : # 시계 방향 회전
                data[i].rotate(k)
            else : # 반시계 방향 회전
                data[i].rotate(-k)

    # 원판에 수가 남아 있으면, 인접하면서 수가 같은 것을 제외
    if result != 0 :
        remove_value = []
        for i in range(n) :
            for j in range(m-1) :
                if data[i][j] != 0 and data[i][j+1] != 0 and data[i][j] == data[i][j+1] :
                    remove_value.append((i, j))
                    remove_value.append((i, j+1))
            if data[i][0] != 0 and data[i][-1] != 0 and data[i][0] == data[i][-1] :
                remove_value.append((i, 0))
                remove_value.append((i, m-1))

        for j in range(m) :
            for i in range(n-1) :
                if data[i][j] != 0 and data[i+1][j] != 0 and data[i][j] == data[i+1][j] :
                    remove_value.append((i, j))
                    remove_value.append((i+1, j))

        # 원판 갱신
        remove_value = list(set(remove_value))
        for i in range(len(remove_value)) :
            x, y = remove_value[i]
            data[x][y] = 0

        # 없는 경우 : 원판에 적힌 수의 평균을 구하고, 평균보다 큰 수에서 1을 빼고 작은 수에는 1을 더한다.
        if len(remove_value) == 0 :
            sum_value = 0
            zero_count = 0
            for i in range(n) :
                sum_value += sum(data[i])
                zero_count += data[i].count(0)
            avg_value = sum_value / (n * m - zero_count)

            for i in range(n) :
                for j in range(m) :
                    if data[i][j] != 0 and data[i][j] > avg_value :
                        data[i][j] -= 1
                    elif data[i][j] != 0 and data[i][j] < avg_value :
                        data[i][j] += 1
    else :
        break

answer = 0
for i in range(n) :
    answer += sum(data[i])

print(answer)

#20061번 모노미노도미노2
n = int(input())
blue = [[0] * 6 for _ in range(4)]
green = [[0] * 4 for _ in range(6)]

result = 0

def move_blue(t, x) :
    global blue
    y = 1
    if t == 1 or t == 2 :
        while True :
            if y + 1 > 5 or blue[x][y + 1] : # 범위를 벗어나거나 블록이 있다면
                blue[x][y] = 1
                if t == 2 :
                    blue[x][y-1] = 1
                break
            y += 1
    else :
        while True :
            if y + 1 > 5 or blue[x][y+1] != 0 or blue[x+1][y+1] != 0 :
                blue[x][y], blue[x+1][y] = 1, 1
                break
            y += 1

    check_blue()

    for j in range(2) :
        for k in range(4) :
            if blue[k][j] :
                remove_blue(5)
                break

def check_blue() :
    global result
    for j in range(2, 6) :
        cnt = 0
        for k in range(4) :
            if blue[k][j] :
                cnt += 1

        if cnt == 4 :
            remove_blue(j)
            result += 1

def remove_blue(index) :
    for j in range(index, 0, -1) :
        for k in range(4) :
            blue[k][j] = blue[k][j-1]
    for j in range(4) :
        blue[j][0] = 0

def move_green(t, y) :
    global green
    x = 1
    if t == 1 or t == 3 :
        while True :
            if x + 1 > 5 or green[x+1][y] :
                green[x][y] = 1
                if t == 3 :
                    green[x-1][y] = 1
                break
            x += 1
    else :
        while True :
            if x + 1 > 5 or green[x+1][y] or green[x+1][y+1] :
                green[x][y], green[x][y+1] = 1, 1
                break
            x += 1

    check_green()

    for j in range(2) :
        for k in range(4) :
            if green[j][k] :
                remove_green(5)
                break

def check_green() :
    global result
    for j in range(2, 6) :
        cnt = 0
        for k in range(4) :
            if green[j][k] :
                cnt += 1

        if cnt == 4 :
            remove_green(j)
            result += 1

def remove_green(index) :
    for j in range(index, 0, -1) :
        for k in range(4) :
            green[j][k] = green[j-1][k]
    for j in range(4) :
        green[0][j] = 0

for _ in range(n) :
    t, x, y = map(int, input().split())
    move_blue(t, x)
    move_green(t, y)

blue_count, green_count = 0, 0
for i in range(4) :
    for j in range(2, 6) :
        if blue[i][j] : # 블록이 존재하면
            blue_count += 1

for i in range(2, 6) :
    for j in range(4) :
        if green[i][j] : # 블록이 존재하면
            green_count += 1

print(result)
print(blue_count + green_count)