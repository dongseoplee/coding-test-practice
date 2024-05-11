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

#19236번 청소년 상어
import sys
import copy
board = [[] for _ in range(4)]
dx = [-1, -1, 0, 1, 1, 1, 0, -1]
dy = [0, -1, -1, -1, 0, 1, 1, 1]
for i in range(4):
    data = list(map(int, sys.stdin.readline().split()))
    fish = []
    for j in range(4):
        fish.append([data[2*j], data[2*j+1] - 1])
    board[i] = fish

res_score = 0
# print(board)

def dfs(sx, sy, score, board):
    global res_score
    score += board[sx][sy][0]
    res_score = max(res_score, score)
    board[sx][sy][0] = 0
    #물고기 이동
    for f in range(1, 17):
        fx, fy = -1, -1
        for x in range(4):
            for y in range(4):
                if board[x][y][0] == f:
                    fx, fy = x, y
                    break
        if fx == -1 and fy == -1:
            continue
        fd = board[fx][fy][1]
        for i in range(8):
            nd = (fd + i) % 8
            nx = fx + dx[nd]
            ny = fy + dy[nd]
            if not (0 <= nx < 4 and 0 <= ny < 4) or (nx == sx and ny == sy):
                continue
            board[fx][fy][1] = nd
            board[fx][fy], board[nx][ny] = board[nx][ny], board[fx][fy]
            break
        #상어 이동
    sd = board[sx][sy][1]
    for i in range(1, 5):
        nx = sx + dx[sd]*i
        ny = sy + dy[sd]*i
        if (0 <= nx < 4 and 0 <= ny < 4) and board[nx][ny][0] > 0:
            dfs(nx, ny, score, copy.deepcopy(board))

dfs(0, 0, 0, board)
print(res_score)

#20058번 마법사 상어와 파이어스톰
import sys
from collections import deque
n, q = map(int, sys.stdin.readline().split())
n = 2**n
arr = [[0]*(n+2)] + [[0] + list(map(int, sys.stdin.readline().split())) + [0] for _ in range(n)] + [[0]*(n+2)]
lst = list(map(int, sys.stdin.readline().split()))
# print(arr)
for L in lst: #Q 시전하는 것을 for문 돌린다.
    L = 2**L

    #1. 부분 회전
    new = [[0 for _ in range(n+2)] for _ in range(n+2)]
    for si in range(1, n+1, L):
        for sj in range(1, n+1, L):
            # print(si, sj)
            for i in range(L):
                for j in range(L):
                    # print(si+i, sj+j)
                    new[si+i][sj+j] = arr[si+L-1-j][sj+i]
            # print("-")
    arr = new
    #2. 상하좌우 0 2개이상이면 -1
    new = [x[:] for x in arr]
    for i in range(1, n+1):
        for j in range(1, n+1):
            if arr[i][j] == 0: continue
            cnt = 0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = i+dx, j+dy
                if arr[nx][ny] == 0:
                    cnt += 1
                    if cnt >= 2:
                        new[i][j] -= 1
                        break
    arr = new

#3. BFS로 얼음덩어리 크기, visited 사용

def bfs(x, y):
    queue = deque()
    queue.append((x, y))
    visited[x][y] = True
    cnt = 1
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx = x + dx
            ny = y + dy
            if visited[nx][ny] == False and arr[nx][ny] != 0:
                queue.append((nx, ny))
                visited[nx][ny] = True
                cnt += 1
    return cnt



visited = [[False for _ in range(n+2)] for _ in range(n+2)]
res = 0
for i in range(1, n+1):
    for j in range(1, n+1):
        if visited[i][j] == False and arr[i][j] != 0:
            res = max(res, bfs(i, j))
print(sum(map(sum, arr)))
print(res)

#20057번 마법사 상어와 토네이도
# 토네이도 이동 좌표 연습
# 7x7
ci, cj = 7 // 2, 7 //2
#3,3 -> 좌, 하, 우 ,상
dx = [0, 1, 0, -1]
dy = [-1, 0, 1, 0]

flag = 0
idx = 0
maxcnt= 0
print(ci, cj)
while (ci, cj) != (0, 0):
    if flag % 2 == 0:
        maxcnt += 1
    for _ in range(maxcnt):
        ci += dx[idx%4]
        cj += dy[idx%4]
        # print(idx)
        print(ci, cj)
    idx += 1
    flag += 1


#17142번 연구소3
import sys
from collections import deque
# sys.stdin = open("input.txt", "r")

def bfs(tlst):
    queue = deque()
    visited = [[False for _ in range(n)] for _ in range(n)]
    for ti, tj in tlst:
        queue.append((ti, tj))
        visited[ti][tj] = 1
    cnt = CNT
    while queue:
        ci, cj = queue.popleft()
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = ci + di, cj + dj
            if 0 <= ni < n and 0 <= nj < n and visited[ni][nj] == False and arr[ni][nj] != 1:
                queue.append((ni, nj))
                visited[ni][nj] = visited[ci][cj] + 1
                if arr[ni][nj] == 0:
                    cnt -= 1
                    if cnt == 0:
                        return visited[ni][nj] - 1
    return n*n


def dfs(a, s, tlst):
    global ans
    # 종료조건
    if a == m:
        ans = min(ans, bfs(tlst))
        return
    #하부 함수 호출
    for j in range(s, VCNT):
        dfs(a+1, j+1, tlst +[vlst[j]])

#빈칸 갯수(CNT), 바이러스 좌표(vlst)
n, m = map(int, sys.stdin.readline().split())
arr = []
for _ in range(n):
    arr.append(list(map(int, sys.stdin.readline().split())))

#1. 입력 처리 받기
CNT = 0
vlst = []
for i in range(n):
    for j in range(n):
        if arr[i][j] == 0:
            CNT += 1
        if arr[i][j] == 2:
            vlst.append((i, j))
VCNT = len(vlst) # 바이러스 갯수

#2. m개 선택 (조합) 후 최소값 갱신, 백트래킹




if CNT == 0:
    ans = 0
else:
    ans = n*n
    dfs(0, 0, [])
    if ans == n*n:
        ans = -1
print(ans)

#12100 2048 (Easy)
import sys
# sys.stdin = open("input.txt", "r")
# 최대 5번 이동 상하좌우 -> DFS, 총 1024가지 계산 (백트래킹).
# 판 기울임은 왼쪽으로 기울이는 것 하나만 만들고.
# 우, 상, 하에 대해서는 배열의 구조를 바꿔서 왼쪽을 기울인 값이 우, 상, 하로 기울인 값으로 되게 진행!!!
N = int(sys.stdin.readline())
arr = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
ans = 0
def move(arr):
    for i in range(len(arr)):
        num = 0
        tlst = []
        for n in arr[i]:
            if n==0:
                continue
            if n == num:
                tlst.append(num*2)
                num = 0
            else:
                if num == 0:
                    num = n
                else:
                    tlst.append(num)
                    num = n
        if num > 0:
            tlst.append(num)
        arr[i] = tlst + [0]*(N-len(tlst))

def dfs(n, arr):
    global ans
    if n == 5:
        ans = max(ans, max(map(max, arr))) # 2차원 리스트에서의 최댓값!!
        return
    narr = [lst[:] for lst in arr] #딥 카피
    move(narr)
    dfs(n+1, narr)

    narr = [lst[::-1] for lst in arr] #딥 카피
    move(narr)
    dfs(n+1, narr)

    arr_t = list(map(list, zip(*arr))) #열을 행으로 바꾸기!!
    narr = [lst[:] for lst in arr_t]
    move(narr)
    dfs(n+1, narr)

    narr = [lst[::-1] for lst in arr_t]
    move(narr)
    dfs(n+1, narr)

dfs(0, arr)
print(ans)


#17837번 새로운 게임 2
import sys
# sys.stdin = open("input.txt", "r")
N, K = map(int, sys.stdin.readline().split())
arr = [[2]*(N+2)]
for _ in range(N):
    arr.append([2] + list(map(int, sys.stdin.readline().split())) + [2])
arr.append([2]*(N+2))
lst = []
v = [[[] for _ in range(N+2)] for _ in range(N+2)]
for k in range(K):
    i, j, d = list(map(int, sys.stdin.readline().split()))
    lst.append([i, j, d])
    v[i][j].append(k)
di = [0, 0, 0, -1, 1]
dj = [0, 1, -1, 0, 0]
opp_dr = {1:2, 2:1, 3:4, 4:3}
def solve():
    #for, else 구문
    for ans in range(1, 1001):
        #[1] 파란색
        for i in range(K):
            ci, cj, dr = lst[i]
            ni, nj = ci + di[dr], cj + dj[dr]
            if arr[ni][nj] == 2: # 이동 위치가 파란색 -> 방향 반대하고 한칸이동
                dr = opp_dr[dr]
                ni, nj = ci + di[dr], cj + dj[dr]
                lst[i][2] = dr # 방향 업데이트 해주기
                if arr[ni][nj] == 2: # 갈 위치도 파란색 -> 가면 안됨
                    continue #다음 말로
            #[2] 흰색, 빨간색
            for idx in range(len(v[ci][cj])):
                if v[ci][cj][idx] == i: #리스트에서 현재 말 번호 찾음
                    mlst = v[ci][cj][idx:] #내 말 위에 있는 이동시킬 말들
                    if arr[ni][nj] == 1: # 빨간
                        mlst = mlst[::-1]
                    v[ni][nj] += mlst
                    if len(v[ni][nj]) >= 4: #정답 처리, 종료조건
                        return ans
                    v[ci][cj] = v[ci][cj][:idx] # 현재 위치 말 제거

                    for j in mlst: # 이동시킨 번호
                        lst[j][0], lst[j][1] = ni, nj
                    break
    else:
        return -1

ans = solve()
print(ans)

#17825번 주사위 윷놀이
import sys
# sys.stdin = open("input.txt", "r")
lst = list(map(int, sys.stdin.readline().split()))
v = [0] * 4 # 말의 현재 위치 저장
ans = 0
#연결된 노드 번호
#       0   1   2      3    4       5     6     7   8   9    10         11    12    13    14    15      16      17     18   19      20   21   22    23    24    25     26    27   28     29    30    31    32     33    34    35
adj = [[1], [2], [3], [4], [5], [6, 21], [7], [8], [9], [10], [11, 27], [12], [13], [14], [15], [16, 29], [17], [18], [19], [20], [32], [22], [23], [24], [25], [26], [20], [28], [24], [30], [31], [24], [32], [32], [32], [32]]
score = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 13, 16, 19, 25, 30, 35, 22, 24, 28, 27, 26, 0]
def dfs(n, sm):
    global ans
    # [1] 종료 조건
    if n == 10:
        ans = max(ans, sm)
        return
    # [2] 하부 함수 호출
    for j in range(4): # 말 4개 선택
        s = v[j] #j번 말의 현재위치
        c = adj[s][-1] # j번 말 한칸 이동 # 갈림길 판단 위함.
        for _ in range(1, lst[n]): # 나머지 칸 이동
            c = adj[c][0]
        if c == 32 or c not in v:         # 목적지 or 다른 말이 없는 경우 이동가능
            v[j] = c
            dfs(n+1, sm + score[c])
            v[j] = s # dfs 끝나면 다시 원래자리로 원상복구 해줘야함.!!

dfs(0, 0)
print(ans)

#17822번 원판 돌리기
import sys
from collections import deque
# sys.stdin = open("input.txt", "r")

N, M, T = map(int, sys.stdin.readline().split())
arr = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
# print(arr)
def bfs(si, sj):
    q = deque()
    q.append((si, sj))
    v[si][sj] = 1
    cnt = 1
    while q:
        ci, cj = q.popleft()
        for di, dj in ((-1,0), (1,0), (0,-1), (0,1)):                 #네 방향, 미방문, 값이 같으면
            ni, nj = ci + di, (cj + dj)%M               # 2차원 리스트 양 끝 idx 이어졌을 때 처리법
            if 0 <= ni < N and v[ni][nj] == 0 and arr[ci][cj] == arr[ni][nj]:
                q.append((ni, nj))
                v[ni][nj] = 1
                cnt += 1
    if cnt == 1:
        v[si][sj] = 0

for _ in range(T):
    x, d, k = map(int, sys.stdin.readline().split())
    # [1] x의 배수 (arr에서는 x-1) d가 0이면 시계방향, K칸 회전
    for i in range(x-1, N, x):      # x의 배수자리 회전
        if d == 0:
            arr[i] = arr[i][-k:] + arr[i][:-k]
        else:
            arr[i] = arr[i][k:] + arr[i][:k]
    # [2] 인접하며 같은 숫자 표시 v[]
    v = [[0] * M for _ in range(N)]
    for i in range(N):
        for j in range(M):      # 전체 순회하면서 미방문, >0
            if v[i][j] == 0 and arr[i][j] > 0:
                bfs(i, j)       # (i, j)와 인접한 값 표시

    # [2-1] v표시 있으면 모두 지움
    del_flag, sm, cnt = 0, 0, 0
    for i in range(N):
        for j in range(M):
            if v[i][j] == 1:        # 인접한 같은 값 있음
                arr[i][j] = 0
                del_flag = 1
            else:                   # 표시안된 값
                if arr[i][j] > 0:
                    sm += arr[i][j]
                    cnt += 1
    # [2-2] 없으면 평균 구해서 크면 -1, 작으면 +1
    if del_flag == 0 and cnt > 0:
        avg = sm / cnt
        for i in range(N):
            for j in range(M):
                if avg < arr[i][j]:
                    arr[i][j] -= 1
                elif 0 < arr[i][j] < avg:
                    arr[i][j] += 1
    if sm == 0:
        break


print(sum(map(sum, arr)))

#20056번 마법사 상어와 파이어볼
import sys
sys.stdin = open("input.txt", "r")
N,M,K = map(int, input().split())
arr = [list(map(int, input().split())) for _ in range(M)]

di,dj = [-1,-1,0,1,1,1,0,-1],[0,1,1,1,0,-1,-1,-1]
for _ in range(K):
    # [0]:i, [1]:j, [2]:질량, [3]:속도, [4]:방향
    # [1] 개체별 이동
    for i in range(len(arr)):
        arr[i][0]=(arr[i][0]+di[arr[i][4]]*arr[i][3])%N+1
        arr[i][1]=(arr[i][1]+dj[arr[i][4]]*arr[i][3])%N+1

    # [2] 전체개체 정렬(좌표기준으로 정렬 => 같은좌표 처리)
    arr.sort(key=lambda x:(x[0],x[1]))
    arr.append([100,100,0,0,0])         # 패딩: 마지막요소 처리를 위한 인덱스
    new=[]

    # [3] 같은좌표 합치고(2개이상) + 나누고(2개이상)=>new에 추가
    i=0
    while i<len(arr)-1:
        si,sj,m,s,d = arr[i]            # 기준좌표
        start = 0                       # 같으면 0,2,4,8
        for j in range(i+1, len(arr)):
            if (si,sj)==(arr[j][0],arr[j][1]):  #  같은좌표
                m += arr[j][2]
                s += arr[j][3]
                if d%2 != arr[j][4]%2:  # 다른방향 start=1
                    start=1
            else:                       # 다른좌표!
                if j-i==1:              # 1개 => 그냥추가
                    new.append(arr[i])
                else:                   # 여러개
                    if m//5>0:          # 나눠도 1이상이면(파이어볼이 있는경우)
                        for dr in range(start,start+8,2):
                            new.append([si,sj,m//5,s//(j-i),dr])
                break
        i=j
    arr=new

ans = 0
for lst in arr:
    ans+=lst[2]
print(ans)

#19237번 어른 상어
import sys
sys.stdin = open("input.txt", "r")
N,M,K = map(int, input().split())
arr = [list(map(int, input().split())) for _ in range(N)]

# [0] 상어 정보 저장 + 초기 냄새 남김
shk = [[0]*4 for _ in range(M)]
v = [[[-1]*2 for _ in range(N)] for _ in range(N)]
for i in range(N):
    for j in range(N):
        if arr[i][j]>0:
            sn = arr[i][j]-1
            shk[sn]=[sn,i,j,0]              # 상어번호,i,j,dr
            v[i][j][0],v[i][j][1] = sn, K   # 상어번호, 냄새남김(초기냄새)

lst = list(map(int, input().split()))
for i in range(M):                  # 상어마리수
    shk[i][3]=lst[i]                # 방향저장

dtbl = [[[0]*4 for _ in range(5)] for _ in range(M)]    # 방향에따른 방향우선순위 룩업테이블 설정
for i in range(M):
    for j in range(1,5):
        dtbl[i][j]=list(map(int,input().split()))

#       상,하,좌,우
di = [0,-1, 1, 0, 0]
dj = [0, 0, 0,-1, 1]

for ans in range(1, 1001):  # 1초~1000초
    # [1] 각 상어를 이동: 현재방향기준, 빈칸->자기냄새
    for i in range(len(shk)):
        sn,si,sj,sd=shk[i]
        for dr in dtbl[sn][sd]:
            ni,nj=si+di[dr],sj+dj[dr]
            # 범위내 냄새가 없는 경우(빈칸 ==-1)
            if 0<=ni<N and 0<=nj<N and v[ni][nj][0]==-1:
                shk[i]=[sn,ni,nj,dr]
                break
        else:               # 빈칸이 없는 경우=>내냄새
            for dr in dtbl[sn][sd]:
                ni,nj=si+di[dr],sj+dj[dr]
                if 0<=ni<N and 0<=nj<N and v[ni][nj][0]==sn:
                    shk[i]=[sn,ni,nj,dr]
                    break

    # [2-1] 각 칸 냄새 -1
    for i in range(N):
        for j in range(N):
            if v[i][j][0]!=-1:      # 빈칸이 아닌경우(냄새있음)
                v[i][j][1]-=1       # 0되면 빈칸으로 처리
                if v[i][j][1]==0:
                    v[i][j][0]=-1

    # [2-2] 낮은번호상어처리(냄새있고, 내냄새 아니면 => 삭제)
    i=0
    while i<len(shk):
        sn,si,sj,sd=shk[i]
        # 냄새있고(==빈칸이 아니고), 내냄새 아니면 !=sn
        if v[si][sj][0]!=-1 and v[si][sj][0]!=sn:
            shk.pop(i)
        else:                       # 빈칸에 내가 처음 또는 내냄새 => 새냄새 뿌림
            v[si][sj]=[sn,K]
            i+=1

    if len(shk)<=1:                 # 1마리 이하면 종료
        break
else:
    ans=-1
print(ans)

# 코드트리 왕실의 기사 대결
#방향: 상 우 하 좌
di = [-1, 0, 1, 0]
dj = [ 0, 1, 0,-1]

N, M, Q = map(int, input().split())
# 벽으로 둘러싸서, 범위체크 안하고, 범위밖으로 밀리지 않게 처리
arr = [[2]*(N+2)]+[[2]+list(map(int, input().split()))+[2] for _ in range(N)]+[[2]*(N+2)]
units = {}
# v = [[0]*(N+2) for _ in range(N+2)] # 디버거로 동작확인용
init_k = [0]*(M+1)
for m in range(1, M+1):
    si,sj,h,w,k=map(int, input().split())
    units[m]=[si,sj,h,w,k]
    init_k[m]=k                 # 초기 체력 저장(ans 처리용)
    # for i in range(si,si+h):    # 디버그용(제출시 삭제 가능)
    #     v[i][sj:sj+w]=[m]*w

def push_unit(start, dr):       # s를 밀고, 연쇄처리..
    q = []                      # push 후보를 저장
    pset = set()                # 이동 기사번호 저장
    damage = [0]*(M+1)          # 각 유닛별 데미지 누적

    q.append(start)             # 초기데이터 append
    pset.add(start)

    while q:
        cur = q.pop(0)          # q에서 데이터 한개 꺼냄
        ci,cj,h,w,k = units[cur]

        # 명령받은 방향진행, 벽이아니면, 겹치는 다른조각이면 => 큐에 삽입
        ni,nj=ci+di[dr], cj+dj[dr]
        for i in range(ni, ni+h):
            for j in range(nj, nj+w):
                if arr[i][j]==2:    # 벽!! => 모두 취소
                    return
                if arr[i][j]==1:    # 함정인 경우
                    damage[cur]+=1  # 데미지 누적

        # 겹치는 다른 유닛있는 경우 큐에 추가(모든 유닛 체크)
        for idx in units:
            if idx in pset: continue    # 이미 움직일 대상이면 체크할 필요없음

            ti,tj,th,tw,tk=units[idx]
            # 겹치는 경우
            if ni<=ti+th-1 and ni+h-1>=ti and tj<=nj+w-1 and nj<=tj+tw-1:
                q.append(idx)
                pset.add(idx)

            # 겹치지 않는 경우 (이 반대가 확실히 겹치는지 따져보고 사용해야 함)
            # if ni>ti+th-1 or ni+h-1<ti or nj+w-1<tj or nj>tj+tw-1:
            #     pass
            # else:
            #     q.append(idx)
            #     pset.add(idx)

            # 상 우 하 좌 (닿는 경우.. 복잡함)
            # if ((ni==ti+th-1 or ni+h-1==ti) and (tj<=nj<tj+tw or tj<=nj+w-1<tj+tw or nj<=tj<nj+w or nj<=tj+tw-1<nj+w)) or \
            #         ((nj==tj+tw-1 or nj+w-1==tj) and (ti<=ni<ti+th or ti<=ni+h-1<ti+th or ni<=ti<ni+h or ni<=ti+th-1<ni+h)):
            #     q.append(idx)
            #     pset.add(idx)

    # 명령 받은 기사는 데미지 입지 않음
    damage[start]=0

    # for idx in pset:
    #     si,sj,h,w,k = units[idx]
    #     for i in range(si, si + h):
    #         v[i][sj:sj + w] = [0] * w  # 기존위치 지우기

    # 이동, 데미지가 체력이상이면 삭제처리
    for idx in pset:
        si,sj,h,w,k = units[idx]

        if k<=damage[idx]:  # 체력보다 더 큰 데미지면 삭제
            units.pop(idx)
        else:
            ni,nj=si+di[dr], sj+dj[dr]
            units[idx]=[ni,nj,h,w,k-damage[idx]]
            # for i in range(ni,ni+h):
            #     v[i][nj:nj+w]=[idx]*w     # 이동위치에 표시

for _ in range(Q):  # 명령 입력받고 처리(있는 유닛만 처리)
    idx, dr = map(int, input().split())
    if idx in units:
        push_unit(idx, dr)      # 명령받은 기사(연쇄적으로 밀기: 벽이 없는 경우)

ans = 0
for idx in units:
    ans += init_k[idx]-units[idx][4]
print(ans)

# 코드트리 루돌프의 반란
N, M, P, C, D = map(int, input().split())
v = [[0]*N for _ in range(N)]

ri, rj = map(lambda x:int(x)-1, input().split())
v[ri][rj]=-1                                # 루돌프표시(-1)

score = [0]*(P+1)
alive = [1]*(P+1)
alive[0] = 0                                # 첫 번째는 없는 산타
wakeup_turn = [1]*(P+1)

santa = [[N]*2 for _ in range(P+1)]         # 빈 자리, 번호 맞추기
for _ in range(1, P + 1):
    n,i,j = map(int, input().split())
    santa[n]=[i-1,j-1]                      # i, j
    v[i-1][j-1] = n

def move_santa(cur,si,sj,di,dj,mul):
    q = [(cur,si,sj,mul)]           # cur번 산타를 si,sj에서 di,dj방향으로 mul칸 이동

    while q:
        cur,ci,cj,mul=q.pop(0)
        # 진행방향 mul칸만큼 이동시켜서 범위내이고 산타있으면 q삽입/범위밖 처리
        ni,nj=ci+di*mul, cj+dj*mul
        if 0<=ni<N and 0<=nj<N:     # 범위내 => 산타 O, X
            if v[ni][nj]==0:        # 빈 칸 => 이동처리
                v[ni][nj]=cur
                santa[cur]=[ni,nj]
                return
            else:                   # 산타 O => 연쇄이동
                q.append((v[ni][nj],ni,nj,1))   # 한칸 이동, v[ni][nj]: 다음 산타번호
                v[ni][nj]=cur
                santa[cur]=[ni,nj]
        else:                       # 범위밖 => 탈락 => 끝
            alive[cur]=0
            return

for turn in range(1, M+1):
    # [0] 모두 탈락 시(alive 모두 0) => break
    if alive.count(1)==0:
        break

    # [1-1] 루돌프 이동: 가장 가까운 산타찾기
    mn = 2*N**2
    for idx in range(1, P+1):
        if alive[idx]==0:   continue    # 타락한 산타 => skip..

        si,sj=santa[idx]
        dist=(ri-si)**2+(rj-sj)**2      # 현재거리
        if mn>dist:
            mn=dist
            mlst=[(si,sj,idx)]          # 최소거리=>새리스트
        elif mn==dist:                  # 같은최소=>추가
            mlst.append((si,sj,idx))
    mlst.sort(reverse=True)             # 행 큰>열 큰
    si,sj,mn_idx = mlst[0]              # 돌격 목표산타!

    # [1-2] 대상 산타 방향으로 루돌프 이동
    rdi = rdj = 0
    if ri>si:   rdi=-1  # 산타가 좌표 작은값 => -1방향 이동
    elif ri<si: rdi=1

    if rj>sj:   rdj=-1
    elif rj<sj: rdj=1

    v[ri][rj]=0             # 루돌프 현재자리 지우기
    ri,rj = ri+rdi, rj+rdj  # 루돌프 이동
    v[ri][rj]=-1            # 이동한 자리에 표시

    # [1-3] 루돌프와 산타가 충돌한 경우 산타 밀리는 처리
    if (ri,rj)==(si,sj):            # 충돌!
        score[mn_idx]+=C            # 산타는 C점 획득
        wakeup_turn[mn_idx]=turn+2  # 깨어날 턴 번호를 저장
        move_santa(mn_idx,si,sj,rdi,rdj,C)  # 산타 C칸이동

    # [2-1] 순서대로 산타이동: 기절하지 않은 경우(산타의 턴 <= turn)
    for idx in range(1, P+1):
        if alive[idx]==0:           continue    # 탈락한 경우 skip
        if wakeup_turn[idx]>turn:   continue    # 깨어날 턴이 아직 안된경우

        si,sj = santa[idx]
        mn_dist = (ri-si)**2 + (rj-sj)**2
        tlst = []
        # 상우하좌 순으로 최소거리 찾기
        for di,dj in ((-1,0),(0,1),(1,0),(0,-1)):
            ni,nj=si+di,sj+dj
            dist = (ri-ni)**2 + (rj-nj)**2
            # 범위내, 산타 없고(<=0),더 짧은 거리인 경우
            if 0<=ni<N and 0<=nj<N and v[ni][nj]<=0 and mn_dist>dist:
                mn_dist = dist
                tlst.append((ni,nj,di,dj))
        if len(tlst)==0:    continue    # 이동할 위치 없음
        ni,nj,di,dj = tlst[-1]          # 마지막에 추가된(더 짧은 거리)

        # [2-2] 루돌프와 충돌시 처리
        if (ri,rj)==(ni,nj):            # 루돌프와 충돌: 반대로 튕겨나감!
            score[idx]+=D
            wakeup_turn[idx]=turn+2
            v[si][sj]=0
            move_santa(idx,ni,nj,-di,-dj,D)
        else:                           # 빈 칸: 좌표갱신, 이동처리
            v[si][sj]=0
            v[ni][nj]=idx
            santa[idx]=[ni,nj]

    # [3] 점수획득: alive 산타는 +1점
    for i in range(1,P+1):
        if alive[i]==1:
            score[i]+=1

print(*score[1:])

# 코드트리 나무박멸
import sys
sys.stdin = open("input.txt", "r")
INF = -10000
N, M, K, C = map(int, sys.stdin.readline().split())
C = -(C+1)
arr = [[INF]*(N+2)] + [[INF] + list(map(int, sys.stdin.readline().split())) + [INF] for _ in range(N)] + [[INF]*(N+2)]
for i in range(1, N+1):
    for j in range(1, N+1):
        if arr[i][j] == -1:
            arr[i][j] = INF

ans = 0
for _ in range(M):
    # [0] 1년 시작 제초제 감소
    for i in range(1, N+1):
        for j in range(1, N+1):
            if arr[i][j] < 0:
                arr[i][j] += 1

    # [1] 나무 인접 네칸 중 빈칸 수 만큼 성장
    narr = [x[:] for x in arr]
    for i in range(1, N+1):
        for j in range(1, N+1):                         # 나무 +1 성장
            if arr[i][j] > 0:
                for ni, nj in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
                    if arr[ni][nj] > 0:
                        narr[i][j] += 1
    arr = narr
    # [2] 인접 빈칸에 번식
    narr = [x[:] for x in arr]
    for i in range(1, N+1):                             # 빈칸 갯수 narr에 넣기
        for j in range(1, N+1):
            if arr[i][j]>0:                             # 나무면 번식
                tlst = []
                for ni, nj in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
                    if arr[ni][nj] == 0:
                        tlst.append((ni, nj))
                if len(tlst) > 0:
                    cnt = arr[i][j] // len(tlst)
                    for ti, tj in tlst:
                        narr[ti][tj] += cnt
    arr = narr
    # [3] 가장 많이 박멸되는 칸
    mx, mx_i, mx_j = 0, 0, 0
    for i in range(1, N+1):
        for j in range(1, N+1):
            if arr[i][j] > 0:
                cnt = arr[i][j]
                for di, dj in ((-1, -1), (1, -1), (-1, 1), (1, 1)):                                # 대각선으로 나아가기
                    for mul in range(1, K+1):
                        ni, nj = i + di*mul, j + dj*mul
                        if arr[ni][nj] <= 0:
                            break
                        else:
                            cnt += arr[ni][nj]
                if mx < cnt:
                    mx, mx_i, mx_j = cnt, i, j
    if mx == 0:
        break
    ans += mx
    # [2] 제초제 뿌릴 나무 선정
    arr[mx_i][mx_j] = C
    for di, dj in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        for mul in range(1, K + 1):
            ni, nj = mx_i + di*mul, mx_j + dj*mul
            if arr[ni][nj] <= 0:
                if C<=arr[ni][nj]:
                    arr[ni][nj]=C
                break
            else:
                arr[ni][nj] = C
print(ans)

# 코드트리 싸움땅
import sys
sys.stdin = open("input.txt", "r")
N, M, K = map(int, sys.stdin.readline().split())
arr = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
gun = [[[] for _ in range(N)] for _ in range(N)]
for i in range(N):
    for j in range(N):
        if arr[i][j] > 0:
            gun[i][j].append(arr[i][j])

arr = [[0]*N for _ in range(N)]
players = {}
opp = {0:2, 2:0, 1:3, 3:1}#0123 -> 상 우 하 좌
di = [-1, 0, 1, 0]
dj = [0, 1, 0, -1]
def leave(num, ci, cj, cd, cp, cg, cs):
    for k in range(4):
        ni, nj = ci+di[(cd+k)%4], cj + dj[(cd+k)%4]
        if 0<=ni<N and 0<=nj<N and arr[ni][nj] == 0:                                     #범위내, 빈자리
            if len(gun[ni][nj])>0:
                cg=max(gun[ni][nj])
                gun[ni][nj].remove(cg)
            arr[ni][nj]=num
            players[num] = [ni, nj, (cd+k)%4, cp, cg, cs]
            return

for m in range(1, M+1):
    i, j, d, p = map(int, sys.stdin.readline().split())
    players[m] = [i-1, j-1, d, p, 0, 0]
    arr[i-1][j-1] = m

for _ in range(K):
    # [1] 플레이어 이동
    for i in players:           #딕셔너리의 key가 불려진다.
        ci, cj, cd, cp, cg, cs = players[i]
        ni, nj = ci + di[cd], cj + dj[cd]
        if ni<0 or ni>=N or nj<0 or nj>=N:
            cd = opp[cd]
            ni, nj = ci + di[cd], cj + dj[cd]
        arr[ci][cj] = 0
        # [1-2] 이동한 위치에 총이 있는 경우
        if arr[ni][nj] == 0:                    # 사람이 없고
            if len(gun[ni][nj]) > 0:            # 총이 있다.
                mx = max(gun[ni][nj])
                if cg < mx:                     # 더 강한 총이 있으면 바꾼다.
                    if cg > 0:                  # 플레이어 총 가지고 있다.
                        gun[ni][nj].append(cg)
                    gun[ni][nj].remove(mx)
                    cg = mx
            arr[ni][nj] = i
            players[i] = [ni, nj, cd, cp, cg, cs]       # 정보갱
        # [1-3] 이동한 위치에 사람이 있어서 싸우는 경우
        else:
            enemy = arr[ni][nj]
            ei, ej, ed, ep, eg, es = players[enemy]
            if cp+cg>ep+eg or ((cp+cg==ep+eg) and cp>ep):
                cs += (cp+cg)-(ep+eg)
                leave(enemy, ni, nj, ed, ep, 0, es)
                if cg < eg:
                    if cg>0:
                        gun[ni][nj].append(cg)
                    cg=eg
                else:
                    if eg>0:
                        gun[ni][nj].append(eg)
                arr[ni][nj] = i
                players[i] = [ni, nj, cd, cp, cg, cs]
            else:
                es += (ep+eg)-(cp+cg)
                leave(i, ni, nj, cd, cp, 0, cs)
                if eg < cg:
                    if eg>0:
                        gun[ni][nj].append(eg)
                    eg = cg
                else:
                    if cg>0:
                        gun[ni][nj].append(eg)
                arr[ni][nj] = enemy
                players[enemy] = [ni, nj, ed, ep, eg, es]
    print(arr)
for i in players:
    print(players[i][5], end=" ")

# 코드트리 루돌프의 반란
import sys
sys.stdin = open("input.txt", "r")

N, M, P, C, D = map(int, sys.stdin.readline().split())
v = [[0]*N for _ in range(N)]
ri, rj = map(int, sys.stdin.readline().split())
ri, rj = ri-1, rj-1
v[ri][rj] = -1

score = [0]*(P+1)
alive = [1]*(P+1)
alive[0]=0
wakeup_turn = [1]*(P+1)
santa = [[N]*2 for _ in range(P+1)]
for _ in range(1, P+1):
    n, i, j = map(int, sys.stdin.readline().split())
    santa[n] = [i-1, j-1]
    v[i-1][j-1] = n

def move_santa(cur, si, sj, di, dj, mul):
    q = [(cur, si, sj, mul)]
    while q:
        cur, ci, cj, mul = q.pop(0)
        ni, nj = ci + di*mul, cj + dj*mul
        if 0<=ni<N and 0<=nj<N:         # 범위 내
            if v[ni][nj] == 0:
                v[ni][nj] = cur
                santa[cur] = [ni, nj]
                return
            else:
                q.append((v[ni][nj], ni, nj, 1))
                v[ni][nj] = cur
                santa[cur] = [ni, nj]
        else:
            alive[cur] = 0
            return


for turn in range(1, M+1):
    # [0] 살아있는 산타 없으면 break
    if alive.count(1) == 0:
        break
    # [1-1] 루돌프 이동 8방향
    mn = 2*(N**2)
    for idx in range(1, P+1):
        if alive[idx] == 0: continue       # 살아있는 산타만 진행
        si, sj = santa[idx]
        dist = (si-ri)**2 + (sj-rj)**2
        if mn > dist:                   # 가장 가까운 산타 위치, 번호 찾기
            mlst = [(si, sj, idx)]
            mn = dist
        elif mn == dist:        # 같을때!
            mlst.append((si, sj, idx))
    mlst.sort(reverse=True)
    si, sj, mn_idx = mlst[0]            # 가장 가까운 산타 찾고 여기 방향으로 돌진
    # [1-2] 루돌프 이동방향 결정 8방향 중 1방향
    rdi, rdj = 0, 0
    if ri > si: rdi = -1

    elif ri < si: rdi = 1

    if rj > sj: rdj = -1

    elif rj < sj: rdj = 1
    v[ri][rj] = 0
    ri, rj = ri+rdi, rj+rdj
    v[ri][rj] = -1
    # [1-2] 루돌프 이동 위치가 산타 위치랑 동일해서 부딪힘
    if (ri, rj) == (si, sj):
        score[mn_idx] += C
        wakeup_turn[mn_idx] = turn + 2
        move_santa(mn_idx, si, sj, rdi, rdj, C)     # 밀려나는 산타 정보, 방향, 칸 수
    # [2] 산타 이동
    for idx in range(1, P+1):
        if wakeup_turn[idx] > turn: continue
        if alive[idx] == 0: continue

        si, sj = santa[idx]                 # 루돌프와 가까워지는 방향으로 이동
        mn_dist = (si-ri)**2 + (sj-rj)**2
        tlst = []
        # [2-1] 상우하좌 순으로 최단거리 찾기
        for di, dj in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            ni, nj = si+di, sj+dj
            dist = (ri-ni)**2 + (rj-nj)**2
            # 범위내, 산타 없고, 짧은거리
            if 0<=ni<N and 0<=nj<N and v[ni][nj] <= 0 and mn_dist > dist:           # 작아야만 추가, 등호 X
                mn_dist = dist
                tlst.append((ni, nj, di, dj))
        if len(tlst) == 0: continue         # 이동할 위치 없음
        ni, nj, di, dj = tlst[-1]           # 산타가 이동할 위치 ni,nj di, dj 방향으로
        # [2-2] 루돌프와 충돌
        if (ri, rj) == (ni, nj):
            score[idx] += D
            wakeup_turn[idx] = turn + 2
            v[si][sj] = 0
            move_santa(idx, ni, nj, -di, -dj, D)           # (-1) 곱하면 반대방향
        else:
            v[si][sj] = 0
            v[ni][nj] = idx
            santa[idx] = [ni, nj]

    # [3] 탈락하지 않은 산타 점수 +1
    for idx in range(1, P+1):
        if alive[idx] == 1:
            score[idx] += 1

print(*score[1:])

# 코드트리 왕실의 기사
import sys
sys.stdin = open("input.txt", "r")
di = [-1, 0, 1, 0]
dj = [0, 1, 0, -1]
N, M, Q = map(int, sys.stdin.readline().split())
arr = [[2]*(N+2)] + [[2] + list(map(int, sys.stdin.readline().split())) + [2] for _ in range(N)] + [[2]*(N+2)]
units = {}              # 딕셔너리
init_k = [0]*(M+1)
for m in range(1, M+1):
    si, sj, h, w, k = map(int, sys.stdin.readline().split())
    units[m] = [si, sj, h, w, k]
    init_k[m] = k           # 초기 체력 저장

def push_unit(start, dr):
    # [1] start 번 기사를 dr 방향으로 밀기
    q = []
    pset = set()
    damage = [0]*(m+1)
    q.append(start)
    pset.add(start)
    while q:
        cur = q.pop(0)
        ci, cj, h, w, k = units[cur]

        # [1-2] 명령받은 방향 벽이 아니고, 겹치는 기사있으면 큐에 삽입
        ni, nj = ci + di[dr], cj + dj[dr]
        for i in range(ni, ni+h):
            for j in range(nj, nj+w):       # 이동할 구역의 좌표들
                if arr[i][j] == 2:          # 벽이라면 모두 이동 취소
                    return
                if arr[i][j] == 1:
                    damage[cur] += 1
        for idx in units:       # 딕셔너리의 key값
            if idx in pset: continue

            ti, tj, th, tw, tk = units[idx]
            if ni<=ti+th-1 and ni+h-1>=ti and tj<=nj+w-1 and nj<=tj+tw-1:
                q.append(idx)
                pset.add(idx)
    damage[start] = 0
    for idx in pset:
        si, sj, h, w, k = units[idx]

        if k <= damage[idx]:
            units.pop(idx)          # 딕셔너리 삭제 pop
        else:
            ni, nj = si + di[dr], sj + dj[dr]
            units[idx] = [ni, nj, h, w, k - damage[idx]]

for _ in range(Q):
    idx, dr = map(int, sys.stdin.readline().split())
    if idx in units:
        push_unit(idx, dr)      # idx번 기사를 dr 방향으로 밀어라

ans = 0
for i in units:
    ans += init_k[i] - units[i][4]
print(ans)

# 코드트리 술래잡기
# import sys
# sys.stdin = open("input.txt", "r")          # 제출할 때 주석처리
N, MM, H, K = map(int, input().split())
arr = []
for _ in range(MM):
    arr.append(list(map(int, input().split())))

tree = set()
for _ in range(H):
    i, j = map(int, input().split())
    tree.add((i, j))

# 좌 우 하 상
di = [0, 0, 1, -1]
dj = [-1, 1, 0, 0]
opp = {0:1, 1:0, 2:3, 3:2}
# 상 우 하 좌
tdi = [-1, 0, 1, 0]
tdj = [0, 1, 0, -1]
mx_cnt, cnt,flag, val = 1, 0, 0, 1
M = (N+1)//2
ti, tj, td = M, M, 0

ans = 0
for k in range(1, K+1):
    # [1] 도망자 이동
    for i in range(len(arr)):
        if abs(arr[i][0] - ti) + abs(arr[i][1] - tj) <= 3:
            ni, nj = arr[i][0] + di[arr[i][2]], arr[i][1] + dj[arr[i][2]]
            if 1 <= ni <= N and 1 <= nj <= N:       # 이동할 위치 범위 내
                if (ni, nj) != (ti, tj):
                    arr[i][0], arr[i][1] = ni, nj
            else:                                   # 범위 밖으로 나간다면
                arr[i][2] = opp[arr[i][2]]          # 방향 반대 저장
                ni, nj = arr[i][0] + di[arr[i][2]], arr[i][1] + dj[arr[i][2]]
                if (ni, nj) != (ti, tj):
                    arr[i][0], arr[i][1] = ni, nj
    # [2] 술래 이동
    cnt += 1
    ti, tj = ti + tdi[td], tj + tdj[td]             # 술래 위치 이동 시킴
    if (ti, tj) == (1, 1):
        mx_cnt, cnt, flag, val = N, 1, 1, -1
        td = 2
    elif (ti, tj) == (M, M):
        mx_cnt, cnt, flag, val = 1, 0, 0, 1
        td = 0
    else:
        if cnt == mx_cnt:           # 방향 변경
            cnt = 0
            td = (td+val)%4
            if flag == 0:
                flag = 1
            else:
                flag = 0
                mx_cnt += val
    # [3] 도망자 잡기
    # [3-1] 술래 현재 방향으로 3칸 확인
    # set에 괄호 3개 쓴다.
    tset = set(((ti, tj), (ti + tdi[td], tj + tdj[td]), (ti + tdi[td]*2, tj + tdj[td]*2)))
    for i in range(len(arr) -1, -1, -1):
        if (arr[i][0], arr[i][1]) in tset and (arr[i][0], arr[i][1]) not in tree:
            arr.pop(i)
            ans += k
    if len(arr) == 0:           # 도망자가 다 잡힌 경우
        break
print(ans)

# 코드트리 꼬리잡기놀이
import sys
sys.stdin = open("input.txt", "r")
from collections import deque
N, M, K = map(int, input().split())
arr = [list(map(int, input().split())) for _ in range(N)]
ans = 0
def bfs(si, sj, team_n):
    q = deque()
    team = deque()
    q.append((si, sj))
    v[si][sj] = 1
    team.append((si, sj))
    arr[si][sj] = team_n
    while q:
        ci, cj = q.popleft()
        # 네방향, 범위내, 미방문, 조건:2, 3
        for di, dj in((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = ci + di, cj + dj
            if 0<=ni<N and 0<=nj<N and v[ni][nj] == 0:
                # 머리->꼬리 순서대로 좌표를 저장해야하기 때문에 이전좌표가 머리가 아니고 이동할 좌표가 꼬리
                if arr[ni][nj] == 2 or ((ci, cj) != (si, sj) and arr[ni][nj] == 3):
                    q.append((ni, nj))
                    v[ni][nj] = 1
                    team.append((ni, nj))
                    arr[ni][nj] = team_n
    teams[team_n] = team        # 딕셔너리에 추가

v = [[0]*N for _ in range(N)]
team_n, teams = 5, {}
for i in range(N):
    for j in range(N):
        if v[i][j] == 0 and arr[i][j] == 1:                # 미방문, 머리
            bfs(i, j, team_n)
            team_n += 1
# 우 상 좌 하
di = [0, -1, 0, 1]
dj = [1, 0, -1, 0]
for k in range(K):
    # [1] 팀 사람들 이동
    for team in teams.values():
        ei, ej = team.pop()             # 꼬리 삭제
        arr[ei][ej] = 4                 # 길로 변경시킴
        si, sj = team[0]                # 머리 좌표
        # 네방향, 범위 내, 4인 값
        for ni, nj in ((si-1, sj), (si+1, sj), (si, sj-1), (si, sj+1)):
            if 0<=ni<N and 0<=nj<N and arr[ni][nj] == 4:
                team.appendleft((ni, nj))       # 꼬리를 떼고, 머리붙여주면 같은 길이됨!!!!!
                arr[ni][nj] = arr[si][sj]
                break
    # [2] 공 던지기
    dr = (k//N)%4           # 공 던지는 방향 우 상 좌 하
    offset = k%N            # 몇번째 칸
    if dr == 0:
        ci, cj = offset, 0
    elif dr == 1:
        ci, cj = N-1, offset
    elif dr == 2:
        ci, cj = N - 1 - offset, N-1
    elif dr == 3:
        ci, cj = 0, N - 1 - offset
    # [3] 점수 계산
    for _ in range(N):
        if 0<=ci<N and 0<=cj<N and arr[ci][cj] > 4:
            team_n = arr[ci][cj]
            ans += (teams[team_n].index((ci, cj)) + 1)**2
            # teams[team_n] = teams[team_n][::-1]        # 이동 방향 반대 전환
            teams[team_n].reverse()
            break
        ci, cj = ci + di[dr], cj + dj[dr]

print(ans)

# 코드트리 포탑 부수기
# import sys
# sys.stdin = open("input.txt", "r")

N, M, K = map(int, input().split())
arr = [list(map(int, input().split())) for _ in range(N)]
turn = [[0]*M for _ in range(N)]

from collections import deque
def bfs(si, sj, ei, ej):
    q = deque()
    v = [[[] for _ in range(M)] for _ in range(N)]
    q.append((si, sj))
    v[si][sj] = (si, sj)
    d = arr[si][sj]

    while q:
        ci, cj = q.popleft()
        if (ci, cj) == (ei, ej):
            arr[ei][ej] = max(0, arr[ei][ej] - d)
            while True:
                ci, cj = v[ci][cj]
                if (ci, cj) == (si, sj):
                    return True
                arr[ci][cj] = max(0, arr[ci][cj] - d//2)
                fset.add((ci, cj))
        for di, dj in ((0,1), (1,0), (0,-1), (-1,0)):       # 우 하 좌 상
            ni, nj = (ci + di)%N, (cj + dj)%M
            if len(v[ni][nj]) == 0 and arr[ni][nj] > 0:
                q.append((ni, nj))
                v[ni][nj] = (ci, cj)

    return False

def bomb(si, sj, ei, ej):
    d = arr[si][sj]
    arr[ei][ej] = max(0, arr[ei][ej]-d)
    for di, dj in ((-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)):
        ni, nj = (ei + di)%N, (ej + dj)%M
        if (ni, nj) != (si, sj):
            arr[ni][nj] = max(0, arr[ni][nj] - d//2)
            fset.add((ni, nj))

for T in range(1, K+1):
    # [1] 공격자 선정
    mn, mx_turn, si, sj = 5001, 0, -1, -1
    for i in range(N):
        for j in range(M):
            if arr[i][j] <= 0: continue
            if (mn > arr[i][j]) or (mn == arr[i][j] and mx_turn < turn[i][j]) \
                or (mn == arr[i][j] and mx_turn == turn[i][j] and i+j > si+sj) \
                or (mn == arr[i][j] and mx_turn == turn[i][j] and i+j == si+sj and j > sj):
                    mn, mx_turn, si, sj = arr[i][j], turn[i][j], i, j
    # [2] 피해자 선정
    mx, mn_turn, ei, ej = 0, T, N, M
    for i in range(N):
        for j in range(M):
            if arr[i][j] <= 0: continue
            if (mx < arr[i][j]) or (mx == arr[i][j] and mn_turn > turn[i][j]) \
                    or (mx == arr[i][j] and mn_turn == turn[i][j] and i + j < ei + ej) \
                    or (mx == arr[i][j] and mn_turn == turn[i][j] and i + j == ei + ej and j < ej):
                mx, mn_turn, ei, ej = arr[i][j], turn[i][j], i, j
    # [3] 공격
    # [3-1] 레이저 공격
    arr[si][sj] += (N+M)
    turn[si][sj] = T
    fset = set()
    fset.add((si, sj))
    fset.add((ei, ej))
    if bfs(si, sj, ei, ej) == False:         # 레이저 경로가 나온다면
        bomb(si, sj, ei, ej)

    # [3-2] 포탄 공격
    for i in range(N):
        for j in range(M):
            if arr[i][j] > 0 and (i, j) not in fset:
                arr[i][j] += 1
    cnt = N*M
    for lst in arr:
        cnt -= lst.count(0)
    if cnt <= 1:
        break

print(max(map(max, arr)))

# SWEA 1945 소인수분해
# import sys
# sys.stdin = open("input.txt", "r")
T = int(input())
for t in range(T):
    print("#{}".format(t+1), end=" ")
    N = int(input())
    lst = [2, 3, 5, 7, 11]
    ans = [0]*5
    idx = 0
    for tnum in lst:
        while N % tnum == 0:
            ans[idx] += 1
            N = N // tnum
        idx += 1

    print(*ans)

# SWEA 5789 현주의 상자 바꾸기
# import sys
# sys.stdin = open("input.txt", "r")
T = int(input())
for t in range(1, T+1):
    N, Q = map(int, input().split())
    # [1] 자료형 선언
    arr = [0]*(N+1)
    # [2] Q번 반복, LR, 반복문 j
    for i in range(1, Q+1):
        L, R = map(int, input().split())
        for j in range(L, R+1):
            arr[j] = i
    print("#{}".format(t), end=" ")
    print(*arr[1:])

# SWEA 6485 삼성시의 버스노선
# import sys
# sys.stdin = open("input.txt", "r")
T = int(input())
for t in range(1, T+1):
    arr = [0]*5001
    ans = []
    N = int(input())
    for _ in range(N):
        A, B = (map(int, input().split()))
        for j in range(A, B+1):
            arr[j] += 1
    P = int(input())
    for _ in range(P):
        idx = int(input())
        ans.append(arr[idx])
    print("#{}".format(t), end=" ")
    print(*ans)

# SWEA 파리 퇴치
# import sys
# sys.stdin = open("input.txt", "r")
T = int(input())
for t in range(1, T+1):
    N, M = map(int, input().split())
    arr = [list(map(int, input().split())) for _ in range(N)]
    mx, sm = 0, 0
    for i in range(N-M+1):
        for j in range(N-M+1):
            sm = 0
            # i, j에서 시작
            for k in range(i, i+M):
                for l in range(j, j+M):
                    sm += arr[k][l]
            mx = max(mx, sm)

    print("#{}".format(t), end=' ')
    print(mx)

# SWEA 파스칼의 삼각형
T = int(input())
for t in range(1, T+1):
    N = int(input())            # 주변 0 두르는 padding
    arr = [[0]*(N+1) for _ in range(N+1)]
    arr[1][1] = 1

    for i in range(2, N+1):               # 2부터 범위 잡자
        for j in range(1, i+1):
            arr[i][j] = arr[i-1][j-1] + arr[i-1][j]

    # 0은 출력하지 않고 숫자 출력 -> 2중 for문 범위까지만.
    print("#{}".format(t))
    for i in range(1, N+1):
        for j in range(1, i+1):
            print(arr[i][j], end= ' ')
        print()

# SWEA 1206 View
# import sys
# sys.stdin = open("input.txt", "r")
for t in range(10):
    N = int(input())
    tower = list(map(int, input().split()))
    ans = 0
    # [1] idx 2 ~ (길이 - 2)
    for i in range(2, N-2):
        temp = []
        for j in (-2, -1, 1, 2):
            temp.append(max(0, tower[i]-tower[i-j]))
        ans += min(temp)

    print("#{}".format(t+1), end=' ')
    print(ans)

# SWEA 1208 Flatten
# import sys
# sys.stdin = open("input.txt", "r")
for t in range(10):
    D = int(input())
    blst = list(map(int, input().split()))
    # [1] 최대 최소 값과 idx 알아내기
    for d in range(D):          # 덤프 횟수
        mx_num, mn_num = max(blst), min(blst)
        mx_idx, mn_idx = blst.index(mx_num), blst.index(mn_num)
        if mx_num == mn_num:        # 최대 최소 같은 평탄화 완료
            break
        blst[mx_idx], blst[mn_idx] = blst[mx_idx] - 1, blst[mn_idx] + 1
    ans = max(blst) - min(blst)


    print("#{}".format(t+1), end=' ')
    print(ans)

# SWEA 1210 ladder1
# import sys
# sys.stdin = open("input.txt", "r")
T = 10
for test_case in range(1, T+1):
    _ = int(input())
    arr = [[0] + list(map(int, input().split())) + [0] for _ in range(100)]
    ci, cj = 0, 0
    for j in range(1, 101):         # 좌, 우 padding으로 0,1,2~100,101
        if arr[99][j] == 2:
            ci, cj = 99, j
            break
    # print(ci, cj)
    while ci != 0:          # 첫번째 행이 될때까지 계속 올라옴
        if arr[ci][cj-1] == 1:
            arr[ci][cj] = 0
            ci, cj = ci, cj-1

        elif arr[ci][cj+1] == 1:                        # 좌, 우로 갈 수 있다면 이동
            arr[ci][cj] = 0
            ci, cj = ci, cj+1
        elif arr[ci-1][cj] == 1:
            arr[ci][cj] = 0
            ci, cj = ci-1, cj                                     # 좌, 우 못가면 위로 이동
    print("#{} {}".format(test_case, cj-1))

