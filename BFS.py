#1926번 그림
# BFS 현재 정점에 연결된 가까운 점들부터 탐색, 큐를 이용해 구현
import sys
from collections import deque

n, m = map(int, sys.stdin.readline().split())
graph = []

for _ in range(n):
    tempList = list(map(int, sys.stdin.readline().split()))
    graph.append(tempList)

dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]


def bfs(graph, a, b):
    queue = deque()
    queue.append((a, b))  # 큐에 삽입
    graph[a][b] = 0  # 0으로 값 변경 함으로써 방문 처리함

    count = 1  # 넓이 1 먹고 시작함
    while queue:  # 큐 빌때까지 진행
        x, y = queue.popleft()
        for i in range(4):  # 상하좌우 확인
            nx = x + dx[i]
            ny = y + dy[i]
            if nx < 0 or nx >= n or ny < 0 or ny >= m:
                continue
            if graph[nx][ny] == 1:  # 상하좌우에 방문해야할 노드가 있다면
                graph[nx][ny] = 0  # 방문 처리 해주고
                queue.append((nx, ny))  # 큐에 삽입
                count += 1  # 넓이 1 증가

    return count


size = []
for i in range(n):
    for j in range(m):
        if graph[i][j] == 1:
            size.append(bfs(graph, i, j))

if len(size) == 0:
    print(len(size))
    print(0)
else:
    print(len(size))
    print(max(size))


#2178번
import sys
from collections import deque

graph = []
n, m = map(int, sys.stdin.readline().split())
for _ in range(n):
    tempList = []
    tempList = list(map(int, sys.stdin.readline().rstrip()))
    graph.append(tempList)

# 배열에서 우, 좌, 하 ,상 dx는 위 아래 -> row값 dy는 좌 우 -> col값
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]


def bfs(graph):
    queue = deque()
    queue.append((0, 0))

    while queue:
        x, y = queue.popleft()
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if nx < 0 or nx >= n or ny < 0 or ny >= m:
                continue

            try:  # 방문한값 전 값에 +1
                if graph[nx][ny] == 1:
                    queue.append((nx, ny))
                    graph[nx][ny] = graph[x][y] + 1 # 방문한값 전 값에 +1

            except:
                print('error')


bfs(graph)
print(graph[n - 1][m - 1])

#7576번 토마토
# 동시 출발점이 2개인 문제
# 초기에 1인 row, col 큐에 넣어둬라
import sys
from collections import deque

# 1 익은, 0 안익은, -1 없음

m, n = map(int, sys.stdin.readline().split())
graph = []

for _ in range(n):
    tempList = list(map(int, sys.stdin.readline().split()))
    graph.append(tempList)

queue = deque()  # 큐를 bfs 함수 밖에서 선언

for i in range(n):
    for j in range(m):
        if graph[i][j] == 1:
            queue.append((i, j))  # 큐에 바로 삽입


# print(graph)
def bfs(bfs_graph):
    # for i in range(len(ripeList)):
    # queue.append(ripeList.pop())
    # print('queue', queue)
    dy = [-1, 1, 0, 0]
    dx = [0, 0, -1, 1]

    while queue:
        x, y = queue.popleft()
        # print(x, y)
        for i in range(4):
            nx = x + dx[i]  # 좌우 열의 갯수 col
            ny = y + dy[i]  # 위아래 행의 갯수 row
            if nx < 0 or nx >= n or ny < 0 or ny >= m:
                continue

            if bfs_graph[nx][ny] == 0:
                # print(nx, ny)
                queue.append((nx, ny))
                # dist += 1
                bfs_graph[nx][ny] = bfs_graph[x][y] + 1

    return bfs_graph


# 1의 row, col을 리스트 형태로 bfs 함수로 전달
# 1이면 bfs 돌려라
res_graph = bfs(graph)

res = 0
for line in res_graph:
    for tomato in line:
        if tomato == 0:
            print(-1)
            exit()  # 프로그램 종료

    res = max(res, max(line))

print(res - 1)

# 큐를 밖에서 선언 후 바로 1인 row, col append 하는 것
# 결과 그래프를 이중 for문으로 행별로 읽고 행의 값을 판별 하는 것

#2606번 바이러스
import sys
from collections import deque

n = int(sys.stdin.readline())
pairNum = int(sys.stdin.readline())
pList = []
aList = []
bList = []
for _ in range(pairNum):
  a, b = map(int, sys.stdin.readline().split())
  aList.append(a)
  bList.append(b)
visited = []
# 1의 pair를 큐에 넣기
# 큐에 popleft한것의 pair를 visited에 없으면 다시 큐에 넣기,
queue = deque()
for i in range(pairNum):
  if aList[i] == 1:
    queue.append(bList[i])


def bfs():
  while queue:
    x = queue.popleft()
    visited.append(x)
    for i in range(pairNum):
      #양방향으로 생각해야 됨 -> if문 2개
      if aList[i] == x and bList[i] not in visited:
        queue.append(bList[i])
      if bList[i] == x and aList[i] not in visited:
        queue.append(aList[i])


bfs()
# print(visited)
setVisited = set(visited)
print(len(setVisited) - 1)

#1012번 유기농 배추
import sys
from collections import deque

testNum = int(sys.stdin.readline())
for _ in range(testNum):
    m, n, k = map(int, sys.stdin.readline().split())
    graph = [[0 for _ in range(n)] for _ in range(m)]
    for _ in range(k):
        row, col = map(int, sys.stdin.readline().split())
        graph[row][col] = 1
    # print(graph)

    queue = deque()

    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]


    def bfs(i, j):
        queue.append((i, j))
        while queue:
            x, y = queue.popleft()
            for i in range(4):
                nx = x + dx[i]  # row n
                ny = y + dy[i]  # col m
                if nx < 0 or nx >= m or ny < 0 or ny >= n:
                    continue

                if graph[nx][ny] == 1:
                    queue.append((nx, ny))
                    graph[nx][ny] = 2

        return 1  # bfs결과 배추가 단독으로 1개 있는 것은 값이 1로 됨


    # m:5 n:3
    cnt = 0
    for i in range(m):
        for j in range(n):
            if graph[i][j] == 1:
                cnt += bfs(i, j)  # bfs의 리턴 값을 1로 하여 cnt + 하는 방법

    print(cnt)

#7562번 나이트의 이동
import sys
from collections import deque

testNum = int(sys.stdin.readline())


def bfs(length, sx, sy, fx, fy):
    dx = [-1, -2, -2, -1, 1, 2, 2, 1]
    dy = [2, 1, -1, -2, -2, -1, 1, 2]
    visited = set()
    graph = [[0 for _ in range(length)] for _ in range(length)]
    queue = deque()
    queue.append((sx, sy))
    while queue:
        ox, oy = queue.popleft()
        visited.add((ox, oy))
        for i in range(8):
            nx = ox + dx[i]
            ny = oy + dy[i]

            if nx < 0 or nx >= length or ny < 0 or ny >= length or (nx, ny) in visited:
                continue

            graph[nx][ny] = graph[ox][oy] + 1
            visited.add((nx, ny))
            queue.append((nx, ny))

    print(graph[fx][fy])


for _ in range(testNum):
    length = int(sys.stdin.readline())
    sx, sy = map(int, sys.stdin.readline().split())
    fx, fy = map(int, sys.stdin.readline().split())
    bfs(length, sx, sy, fx, fy)

#2583번 영역 구하기
import sys
from collections import deque

queue = deque()
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]


def bfs(a, b):
    cnt = 0
    queue.append((a, b))
    visited[a][b] = True
    while queue:
        x, y = queue.popleft()
        cnt += 1
        # print(x, y)
        for ai in range(4):
            nx = x + dx[ai]
            ny = y + dy[ai]
            if nx < 0 or nx >= m or ny < 0 or ny >= n:
                continue
            if visited[nx][ny] == True:
                continue
            visited[nx][ny] = True
            queue.append((nx, ny))

    return cnt


m, n, k = map(int, sys.stdin.readline().split())
# print(m,n,k)
graph = [[0 for _ in range(n)] for _ in range(m)]
# bfs visited 배열 쓰던 안쓰던 만들고 시작
visited = [[False for _ in range(n)] for _ in range(m)]  # m행 n열
for _ in range(k):
    x1, y1, x2, y2 = map(int, sys.stdin.readline().split())
    for i in range(y1, y2):
        for j in range(x1, x2):
            graph[i][j] = 1
            visited[i][j] = True

resList = []
for a in range(m):
    for b in range(n):
        if graph[a][b] == 0 and visited[a][b] == False:
            resList.append(bfs(a, b))

resList = sorted(resList)
print(len(resList))
for resListData in resList:
    print(resListData, end=' ')

#2667번 단지번호붙이기
import sys
from collections import deque

n = int(sys.stdin.readline())
graph = []
visited = [[False for _ in range(n)] for _ in range(n)]
for _ in range(n):
    tempList = list(map(int, sys.stdin.readline().rstrip()))
    graph.append(tempList)

queue = deque()
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]


def bfs(i, j):
    count = 0
    queue.append((i, j))
    graph[i][j] = 0
    while queue:
        a, b = queue.popleft()
        count += 1
        # print(a, b)
        for k in range(4):
            nx = a + dx[k]
            ny = b + dy[k]
            if nx < 0 or nx >= n or ny < 0 or ny >= n:
                continue
            if graph[nx][ny] == 1:
                graph[nx][ny] = 0
                queue.append((nx, ny))

    return count


total = 0
resList = []
for i in range(n):
    for j in range(n):
        if graph[i][j] == 1:
            resList.append(bfs(i, j))
            total += 1

print(total)
for x in sorted(resList):
    print(x)

#5014번 스타트링크
import sys
from collections import deque
queue = deque()
f, s, g, u, d = map(int, sys.stdin.readline().split())
if s == g:
  print(0)
  exit()
arr = [0 for _ in range(f+1)]
visited = [False for _ in range(f+1)]

dx=[]
dx.append(u)
dx.append(d*(-1))

def bfs(a):
  queue.append(a)
  visited[a] = True
  while queue:
    x = queue.popleft()
    for i in range(2):
      nx = x + dx[i]
      if nx < 1 or nx > f:
        continue
      if visited[nx] == False:
        queue.append(nx)
        visited[nx] = True
        arr[nx] = arr[x] + 1

bfs(s)
# print(arr)
if arr[g] != 0:
  print(arr[g])

else:
  print('use the stairs')

#2468번 안전영역
import sys
from collections import deque
import copy
n = int(sys.stdin.readline())
graph = []
for _ in range(n):
  graph.append(list(map(int, sys.stdin.readline().split())))
queue = deque()

dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]
def bfs(a, b):
  queue.append((a, b))

  while queue:
    x, y = queue.popleft()
    for e in range(4):
      nx = x + dx[e]
      ny = y + dy[e]
      if nx < 0 or nx >= n or ny < 0 or ny >= n:
        continue
      if tempList[nx][ny] != 0:
        queue.append((nx, ny))
        tempList[nx][ny] = 0


maxNum = max(max(graph))
minNum = min(min(graph))
if minNum == maxNum: #0이건 다른 값이건 모두 같으면 최대 갯수는 1개
  print(1)
  exit()

resList = []

for i in range(minNum, maxNum+1):
  global safeCnt
  tempList = copy.deepcopy(graph)

  for j in range(n):
    for k in range(n):
      if tempList[j][k] <= i:
        tempList[j][k] = 0

  #tempList 가지고 bfs 돌려라
  # print(tempList)
  rescnt = 0
  for p in range(n):
    for q in range(n):
      if tempList[p][q] != 0:
        bfs(p, q)
        rescnt += 1
  resList.append(rescnt)

print(max(resList))

#10026번 적록색약
import sys
from collections import deque

n = int(sys.stdin.readline())
graph = []
visited = [[False for _ in range(n)] for _ in range(n)]
for _ in range(n):
    graph.append(list(sys.stdin.readline().rstrip()))

# print(graph)
# print(visited)
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]
queue = deque()


def bfs(x, y, cnt):
    queue.append((x, y))
    visited[x][y] = cnt

    while queue:
        a, b = queue.popleft()
        for i in range(4):
            nx = a + dx[i]
            ny = b + dy[i]
            if nx < 0 or nx >= n or ny < 0 or ny >= n:
                continue
            if graph[nx][ny] == graph[a][b] and visited[nx][ny] == False:  # 주변과 같고 아직 방문하지 않은 노드
                queue.append((nx, ny))
                visited[nx][ny] = cnt


# 미방문 노드면 bfs 돌아라
cnt = 1
for k in range(n):
    for l in range(n):
        if visited[k][l] == False:
            bfs(k, l, cnt)
            cnt += 1  # 거리, 갯수 출력은 for문 돌릴때 사용한 cnt 값을 사용한다. visited의 max max 사용 안함

# print(graph)
# print(visited)

# print(max(max(visited)), end=' ')

# 적록색약
visited = [[False for _ in range(n)] for _ in range(n)]
# R을 G로 변경
for p in range(n):
    for q in range(n):
        if graph[p][q] == 'G':
            graph[p][q] = 'R'

# print(graph)
# print(visited)
cnt2 = 1
for a1 in range(n):
    for b1 in range(n):
        if visited[a1][b1] == False:
            bfs(a1, b1, cnt2)
            cnt2 += 1
# print(graph)
# print(visited)
# print(max(max(visited)), end=' ')
print(cnt - 1, cnt2 - 1)

#6593번 상범빌딩
import sys
from collections import deque

while True:
    # 동일 명의 변수 선언하지 말자
    dx = [-1, 1, 0, 0, 0, 0]
    dy = [0, 0, -1, 1, 0, 0]
    dz = [0, 0, 0, 0, -1, 1]


    def bfs(z, x, y):
        queue.append((z, x, y))
        visited[z][x][y] = 1
        # print(queue)
        # print(1)
        while queue:
            z, x, y = queue.popleft()
            # print(z, x, y)
            # print(1)
            for k in range(6):
                nz = z + dz[k]
                nx = x + dx[k]
                ny = y + dy[k]
                # print(nz, nx, ny)
                # print(1)
                # print(l, r, c)
                if nz < 0 or nz >= l or nx < 0 or nx >= r or ny < 0 or ny >= c:  # c가 왜 0이지? #동일 명의 변수 선언하지 말자
                    # print(3)
                    continue

                if (visited[nz][nx][ny] == 0 and graph[nz][nx][ny] == '.') or graph[nz][nx][ny] == 'E':
                    # print(2)
                    visited[nz][nx][ny] = visited[z][x][y] + 1
                    queue.append((nz, nx, ny))
                    if graph[nz][nx][ny] == 'E':
                        print('Escaped in', end=' ')
                        print(visited[nz][nx][ny] - 1, end=' ')
                        print('minute(s).')
                        return  # 함수 끝내는 명령어 return
        print('Trapped!')


    l, r, c = map(int, sys.stdin.readline().split())
    if l == 0 and r == 0 and c == 0:
        break
    visited = [[[0 for _ in range(c)] for _ in range(r)] for _ in range(l)]  # 열, 행, 높이
    graph = [[] * r for _ in range(l)]  # 행, 높이로 2차원 배열 만들고 여기에 입력값 넣어서 3차원 배열 만듬
    # print(visited)
    # print(l,r,c)
    for i in range(l):
        for j in range(r):
            graph[i].append(list(map(str, sys.stdin.readline().rstrip())))
        sys.stdin.readline()
    # print(graph)

    queue = deque()

    for a in range(l):
        for b in range(r):
            for d in range(c):
                if graph[a][b][d] == 'S':
                    bfs(a, b, d)

    # print(visited)

#2206번 벽 부수고 이동하기
import sys
from collections import deque

graph = []
n, m = map(int, sys.stdin.readline().split())
for _ in range(n):
    tempList = list(sys.stdin.readline().rstrip())
    graph.append(list(map(int, tempList)))

visited = [[[0] * 2 for _ in range(m)] for _ in range(n)]
visited[0][0][0] = 1
# 3차원 배열 선언법 !!!

dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]


def bfs(x, y, z):
    queue = deque()
    queue.append((x, y, z))

    while queue:
        a, b, c = queue.popleft()
        # 끝 점에 도달하면 이동 횟수를 출력
        if a == n - 1 and b == m - 1:
            return visited[a][b][c]
        for i in range(4):
            nx = a + dx[i]
            ny = b + dy[i]
            if nx < 0 or nx >= n or ny < 0 or ny >= m:
                continue
            # 다음 이동할 곳이 벽이고, 벽파괴기회를 사용하지 않은 경우
            if graph[nx][ny] == 1 and c == 0:
                visited[nx][ny][1] = visited[a][b][0] + 1
                queue.append((nx, ny, 1))
            # 다음 이동할 곳이 벽이 아니고, 아직 한 번도 방문하지 않은 곳이면
            elif graph[nx][ny] == 0 and visited[nx][ny][c] == 0:
                visited[nx][ny][c] = visited[a][b][c] + 1
                queue.append((nx, ny, c))
    return -1


print(bfs(0, 0, 0))

#4963번 섬의 개수

import sys
from collections import deque

while (1):
    h, w = map(int, sys.stdin.readline().split())  # h: 5, w: 4
    if h == 0 and w == 0:
        break
    graph = []
    visited = [[False for _ in range(h)] for _ in range(w)]
    for _ in range(w):
        graph.append(list(map(int, sys.stdin.readline().split())))

    # print(graph)
    # print(visited)
    queue = deque()
    # 상하좌우 대각선
    dx = [-1, 1, 0, 0, -1, -1, 1, 1]  # 행
    dy = [0, 0, 1, -1, -1, 1, -1, 1]  # 열


    def bfs(a, b, cnt):
        queue.append((a, b))
        visited[a][b] = cnt
        while queue:
            x, y = queue.popleft()
            for k in range(8):  # 8번
                nx = x + dx[k]
                ny = y + dy[k]
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue
                if graph[nx][ny] == 1 and visited[nx][ny] == False:
                    queue.append((nx, ny))
                    visited[nx][ny] = cnt


    cnt = 0
    for i in range(w):
        for j in range(h):
            if graph[i][j] == 1 and visited[i][j] == False:
                cnt += 1
                bfs(i, j, cnt)

    # print(visited)
    print(cnt)

#1325번 효율적인 해킹
import sys
from collections import deque

n, m = map(int, sys.stdin.readline().split())
graph = [[] for _ in range(n + 1)]
visited = [False for _ in range(n + 1)]
res = [0 for _ in range(n + 1)]
for _ in range(m):
    a, b = map(int, sys.stdin.readline().split())
    graph[b].append(a)

queue = deque()


def bfs(a):
    queue.append(a)
    visited[a] = True
    cnt = 0
    while queue:
        x = queue.popleft()
        for node in graph[x]:
            if visited[node] == False:
                queue.append(node)
                visited[node] = True
                cnt += 1
    return cnt


for i in range(1, n + 1):
    res[i] = bfs(i)
    visited = [False for _ in range(n + 1)]

# print(res)
for j in range(1, n + 1):
    if res[j] == max(res):
        print(j, end=' ')

#1743번 음식물 피하기
import sys
from collections import deque

n, m, k = map(int, sys.stdin.readline().split())
graph = [[0 for _ in range(m)] for _ in range(n)]
visited = [[False for _ in range(m)] for _ in range(n)]
for _ in range(k):
    a, b = map(int, sys.stdin.readline().split())
    graph[a - 1][b - 1] = 1

# print(graph)
queue = deque()


def bfs(x, y):
    cnt = 0
    queue.append((x, y))
    visited[x][y] = True
    cnt += 1
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    while queue:
        (px, py) = queue.popleft()
        for k in range(4):
            nx = px + dx[k]
            ny = py + dy[k]
            if nx < 0 or nx >= n or ny < 0 or ny >= m:  # continue 조건 적는 걸 까먹음
                continue
            if graph[nx][ny] == 1 and visited[nx][ny] == False:
                queue.append((nx, ny))
                visited[nx][ny] = True
                cnt += 1

    return cnt


trashSize = []
for i in range(n):
    for j in range(m):
        if visited[i][j] == False and graph[i][j] == 1:
            # print(i, j)
            # print(bfs(i, j))
            trashSize.append(bfs(i, j))

# print(trashSize)
print(max(trashSize))

#14502번 연구소
import sys
import copy
from collections import deque

n, m = map(int, sys.stdin.readline().split())
graph = []
visited = [[False for _ in range(m)] for _ in range(n)]
for _ in range(n):
    graph.append(list(map(int, sys.stdin.readline().split())))

dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

max_res = 0


def bfs():
    global max_res  # 함수 밖 max_res를 사용하기위해 global 선언
    temp = copy.deepcopy(graph)
    res = 0
    queue = deque()

    for p in range(n):
        for q in range(m):
            if graph[p][q] == 2:
                queue.append((p, q))

    while queue:
        x, y = queue.popleft()

        for k in range(4):
            nx = x + dx[k]
            ny = y + dy[k]
            if nx < 0 or nx >= n or ny < 0 or ny >= m:
                continue
            if temp[nx][ny] == 0:
                queue.append((nx, ny))
                temp[nx][ny] = 2

    for row in range(n):
        res += temp[row].count(0)
    max_res = max(max_res, res)


def makeWall(cnt):  # 백트래킹
    if cnt == 3:
        bfs()
        # print(graph)
        return
    for i in range(n):
        for j in range(m):
            if graph[i][j] == 0:
                graph[i][j] = 1  # 벽세우기
                makeWall(cnt + 1)
                graph[i][j] = 0  # 벽 허물기


makeWall(0)
print(max_res)

#1389번 케빈 베이컨의 6단계 법칙
import sys
from collections import deque

n, m = map(int, sys.stdin.readline().split())
res = [0] * (n)
graph = [[] for _ in range(n + 1)]
visited = [0 for _ in range(n + 1)]
for _ in range(m):
    a, b = map(int, sys.stdin.readline().split())
    graph[a].append(b)
    graph[b].append(a)

# print(graph)
queue = deque()


def bfs(j):
    while queue:
        a = queue.popleft()
        for queueData in graph[a]:
            if visited[queueData] == 0:
                visited[queueData] = visited[a] + 1
                queue.append(queueData)
    res[j - 1] = sum(visited) - visited[j]


for j in range(1, n + 1):
    for graphIdx1 in graph[j]:
        queue.append(graphIdx1)
        visited[graphIdx1] = 1
        visited[j] = 1

    bfs(j)
    # print(visited)
    visited = [0 for _ in range(n + 1)]

# print(res)
print(res.index(min(res)) + 1)

#11403번 경로 찾기
import sys
from collections import deque

n = int(sys.stdin.readline())
graph = []
for _ in range(n):
    graph.append(list(map(int, sys.stdin.readline().split())))

visited = [[0 for _ in range(n)] for _ in range(n)]
queue = deque()


def bfs(idx, a):
    queue.append(a)
    visited[idx][a] = 1
    while queue:
        x = queue.popleft()
        for i in range(n):
            if graph[x][i] == 1 and visited[idx][i] == 0:
                queue.append(i)
                visited[idx][i] = 1


for j in range(n):
    for k in range(n):
        if graph[j][k] == 1:
            bfs(j, k)

for l in range(n):
    print(*visited[l])
    # visited에서 True인 값

#16234번 인구 이동
import sys
from collections import deque
import math

dx = [1, 0, -1, 0]
dy = [0, 1, 0, -1]

n, l, r = map(int, sys.stdin.readline().split())
arr = list()
a_list = list()
for i in range(n):
    arr.append(list(map(int, sys.stdin.readline().split())))


def bfs(i, j):
    dq = deque()
    dq.append((i, j))
    visit[i][j] = True

    union = [(i, j)]
    count = arr[i][j]
    while dq:
        x, y = dq.popleft()
        for d in range(4):
            nx = x + dx[d]
            ny = y + dy[d]
            if nx < 0 or ny < 0 or nx >= n or ny >= n:
                continue
            if visit[nx][ny]:
                continue
            if l <= abs(arr[nx][ny] - arr[x][y]) <= r:
                union.append((nx, ny))
                visit[nx][ny] = True
                dq.append((nx, ny))
                count += arr[nx][ny]

    for x, y in union:
        arr[x][y] = math.floor(count / len(union))

    return len(union)


result = 0
while True:
    visit = [[False] * n for _ in range(n)]
    flag = False

    for i in range(n):
        for j in range(n):
            if not visit[i][j]:
                if bfs(i, j) > 1:
                    flag = True
    if not flag:
        break
    result += 1

print(result)

#2636번 치즈
import sys
from collections import deque

n, m = map(int, sys.stdin.readline().split())
graph = []
res = 0
for _ in range(n):
    graph.append(list(map(int, sys.stdin.readline().split())))
# print(graph)
visited = [[False for _ in range(m)] for _ in range(n)]
melt = [[False for _ in range(m)] for _ in range(n)]
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]
queue = deque()
meltXYQueue = deque()
meltTimeCnt = []


def bfs(x, y):
    meltCnt = 0
    queue.append((x, y))
    visited[x][y] = True
    while queue:
        a, b = queue.popleft()  # 큐에 넣었으니 popleft를 해야함
        for i in range(4):
            nx = a + dx[i]
            ny = b + dy[i]
            if nx < 0 or nx >= n or ny < 0 or ny >= m:
                continue
            if graph[nx][ny] == 0 and visited[nx][ny] is False:  # 0값으로 BFS 탐색한다.
                queue.append((nx, ny))
                visited[nx][ny] = True
            if graph[nx][ny] == 1 and visited[nx][ny] is False:
                visited[nx][ny] = True
                meltCnt += 1
                meltXYQueue.append((nx, ny))

    meltTimeCnt.append(meltCnt)


while (1):

    bfs(0, 0)
    while meltXYQueue:  # 큐를 이용해 녹은 것 처리 1->0
        a, b = meltXYQueue.popleft()
        graph[a][b] = 0

    visited = [[False for _ in range(m)] for _ in range(n)]
    res += 1
    if meltTimeCnt[res - 1] == 0:  # 녹는 갯수가 0개라고 확인되면 다 녹은것임으로 종료
        break

print(res - 1)
print(meltTimeCnt[res - 2])

#7569번 토마토
import sys  # 출발점이 여러개인 bfs는 출발점을 모두 큐에 넣고 시작한다.
from collections import deque

# 3차원 리스트 입력받기

m, n, h = map(int, sys.stdin.readline().split())
graph = [[list(map(int, sys.stdin.readline().split())) for _ in range(n)] for _ in range(h)]
visited = [[[False] * m for _ in range(n)] for _ in range(h)]

# print(graph)
queue = deque()

for k in range(h):
    for i in range(n):
        for j in range(m):
            if graph[k][i][j] == 1:
                queue.append((k, i, j))
                visited[k][i][j] = True

dy = [0, 0, 0, 0, -1, 1]
dx = [0, 0, -1, 1, 0, 0]
dz = [-1, 1, 0, 0, 0, 0]  # z는 -1, 1이 인덱스 0, 1에 적어야 함


# print(visited)
def bfs():
    while queue:
        z, x, y = queue.popleft()
        for i in range(6):
            nx = x + dx[i]
            ny = y + dy[i]
            nz = z + dz[i]
            if nx < 0 or nx >= n or ny < 0 or ny >= m or nz < 0 or nz >= h:
                continue
            if graph[nz][nx][ny] == 0 and visited[nz][nx][ny] == False:
                queue.append((nz, nx, ny))
                visited[nz][nx][ny] = True
                graph[nz][nx][ny] = graph[z][x][y] + 1


bfs()
# print(graph)
maxDay = 0
for graphHeight in graph:
    for graphRow in graphHeight:
        for graphRowNum in graphRow:
            if graphRowNum >= maxDay:
                maxDay = graphRowNum
            if graphRowNum == 0:
                print(-1)
                exit()

print(maxDay - 1)

#2573번 빙산
import sys
from collections import deque
input = sys.stdin.readline


def bfs(x, y):
    q = deque([(x, y)])
    visited[x][y] = 1
    seaList = []

    while q:
        x, y = q.popleft()
        sea = 0
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if 0 <= nx < n and 0 <= ny < m:
                if not graph[nx][ny]:
                    sea += 1
                elif graph[nx][ny] and not visited[nx][ny]:
                    q.append((nx, ny))
                    visited[nx][ny] = 1
        if sea > 0:
            seaList.append((x, y, sea))
    for x, y, sea in seaList:
        graph[x][y] = max(0, graph[x][y] - sea)

    return 1


n, m = map(int, input().split())
graph = [list(map(int, input().split())) for _ in range(n)]

ice = []
for i in range(n):
    for j in range(m):
        if graph[i][j]:
            ice.append((i, j))

dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]
year = 0

while ice:
    visited = [[0] * m for _ in range(n)]
    delList = []
    group = 0
    for i, j in ice:
        if graph[i][j] and not visited[i][j]:
            group += bfs(i, j)
        if graph[i][j] == 0:
            delList.append((i, j))
    if group > 1:
        print(year)
        break
    ice = sorted(list(set(ice) - set(delList)))
    year += 1

if group < 2:
    print(0)

#2146번 다리 만들기
import sys
from collections import deque

q = deque()
n = int(sys.stdin.readline())
graph = []
visited = [[False for _ in range(n)] for _ in range(n)]
for _ in range(n):
    graph.append(list(map(int, sys.stdin.readline().split())))

# print(graph)
# 섬 번호 매기기
# 바다에 인접한 좌표에서 bfs 탐색한 후 출발 섬이 아니면 리스트에 담기 다음섬이 출발섬이 아니면 저장하고 종료
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]


def bfs(x, y, num):
    q.append((x, y))
    visited[x][y] = True
    graph[x][y] = num
    while q:
        a, b = q.popleft()
        for i in range(4):
            nx = a + dx[i]
            ny = b + dy[i]
            if nx < 0 or nx >= n or ny < 0 or ny >= n:
                continue
            else:
                if graph[nx][ny] == 1 and visited[nx][ny] == False:
                    graph[nx][ny] = num
                    q.append((nx, ny))
                    visited[nx][ny] = True


def bfs2(num):
    global answer
    dist = [[-1 for _ in range(n)] for _ in range(n)]
    q = deque()

    for i in range(n):
        for j in range(n):
            if graph[i][j] == num:  # 본인의 섬이라면 ex) 1번 섬을 이루는 좌표가 모두 큐에 들어감!!
                q.append((i, j))
                dist[i][j] = 0  # 거리는 0
    while q:
        x, y = q.popleft()
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if nx < 0 or nx >= n or ny < 0 or ny >= n:
                continue
            if graph[nx][ny] > 0 and graph[nx][ny] != num:  # 육지이며 본인 섬과 다른 섬
                answer = min(answer, dist[x][y])
            if graph[nx][ny] == 0 and dist[nx][ny] == -1:  # 바다이며 방문하지 않았던 바다
                dist[nx][ny] = dist[x][y] + 1
                q.append((nx, ny))


num = 1
for i in range(n):
    for j in range(n):
        if graph[i][j] == 1 and visited[i][j] == False:
            # print(i, j)
            bfs(i, j, num)
            num += 1

answer = sys.maxsize
for l in range(1, num):
    bfs2(l)

print(answer)

#2468번 안전 영역
import sys
from collections import deque
import copy

n = int(sys.stdin.readline())
visited = [[False for _ in range(n)] for _ in range(n)]
graph = []
min_num = sys.maxsize
max_num = -1
for i in range(n):
    graph.append(list(map(int, sys.stdin.readline().split())))
    for graph_num in graph[i]:
        min_num = min(min_num, graph_num)
        max_num = max(max_num, graph_num)

# print(graph)
# print(min_num)
# print(max_num)
res = [0] * (max_num + 1)


def rain(height):
    tempGraph = copy.deepcopy(graph)
    for i in range(n):
        for j in range(n):
            if tempGraph[i][j] <= height:
                tempGraph[i][j] = 0

    return tempGraph


# print(rain(2))
queue = deque()

dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]


def bfs(x, y, cnt):
    queue.append((x, y))
    visited[x][y] = True
    temp[x][y] = cnt
    while queue:
        a, b = queue.popleft()
        for i in range(4):
            nx = a + dx[i]
            ny = b + dy[i]
            if nx < 0 or nx >= n or ny < 0 or ny >= n:
                continue
            elif temp[nx][ny] != 0 and visited[nx][ny] == False:
                queue.append((nx, ny))
                visited[nx][ny] = True
                temp[nx][ny] = cnt


for l in range(min_num, max_num + 1):
    temp = rain(l)
    visited = [[False for _ in range(n)] for _ in range(n)]
    cnt = 1
    for j in range(n):
        for k in range(n):
            if temp[j][k] != 0 and visited[j][k] == False:
                bfs(j, k, cnt)
                cnt += 1

    res[l] = cnt - 1

if max(res) == 0:
    print(1)
else:
    print(max(res))

#4179번 불!
import sys
from collections import deque
input = sys.stdin.readline

n, m = map(int, input().split())
graph = []

for i in range(n):
    graph.append(list(input().rstrip()))
    if 'J' in graph[i]:
        q = deque([(0, i, graph[i].index('J'))])

for i in range(n):
    for j in range(m):
        if graph[i][j] == 'F':
            q.append((-1, i, j))

dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]
ans = 'IMPOSSIBLE'

while q:
    time, x, y = q.popleft()

    # 지훈이 탈출
    if time > -1 and graph[x][y] != 'F' and (x == 0 or y == 0 or x == n - 1 or y == m - 1):
        ans = time + 1
        break

    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
        if 0 <= nx < n and 0 <= ny < m and graph[nx][ny] != '#':

            # 지훈이 이동
            if time > -1 and graph[nx][ny] == '.':
                graph[nx][ny] = '_'
                q.append((time + 1, nx, ny))

            # 불 퍼뜨리기
            elif time == -1 and graph[nx][ny] != 'F':
                graph[nx][ny] = 'F'
                q.append((-1, nx, ny))

print(ans)

#14940번 쉬운 최단거리
import sys
from collections import deque
n, m = map(int, sys.stdin.readline().split())
graph = []
visited = [[False for _ in range(m)] for _ in range(n)]
dist = [[0 for _ in range(m)] for _ in range(n)]

for _ in range(n):
  graph.append(list(map(int, sys.stdin.readline().split())))
queue = deque()
for i in range(n):
  for j in range(m):
    if graph[i][j] == 2:
      queue.append((i, j))
      visited[i][j] = True
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]
def bfs(graph, visited):
  while queue:
    x, y = queue.popleft()
    for i in range(4):
      nx = x + dx[i]
      ny = y + dy[i]
      if nx < 0 or nx >= n or ny < 0 or ny >= m:
        continue
      elif graph[nx][ny] == 1 and visited[nx][ny] == False:
        queue.append((nx, ny))
        visited[nx][ny] = True
        dist[nx][ny] = dist[x][y] + 1


bfs(graph, visited)
for k in range(n):
  for l in range(m):
    if graph[k][l] != 0 and visited[k][l] == False:
      dist[k][l] = -1

for distRow in dist:
  for distRowData in distRow:
    print(distRowData, end=' ')
  print()
  
#12851번 숨바꼭질2
import sys
from collections import deque

N, K = map(int, sys.stdin.readline().split())
queue = deque()
queue.append(N)
way = [0] * 100001
cnt, result = 0, 0
while queue:
    a = queue.popleft()
    temp = way[a]
    if a == K:  # 둘이 만났을 때
        result = temp  # 결과
        cnt += 1  # 방문 횟수 +1
        continue

    for i in [a - 1, a + 1, a * 2]:
        if 0 <= i < 100001 and (way[i] == 0 or way[i] == way[a] + 1):
            way[i] = way[a] + 1
            queue.append(i)
print(result)
print(cnt)

#1600번 말이 되고픈 원숭이
from collections import deque

# 입력값
k = int(input())
m, n = map(int, input().split())
graph = [list(map(int, input().split())) for _ in range(n)]

# 상하좌우, 체스 나이트 이동
dist = [[1,0],[0,1],[-1,0],[0,-1]]
horse = [[-2,-1], [-2,1],[-1,-2],[-1,2],[2,-1],[2,1],[1,-2],[1,2]]

def bfs():
  visited = [[[0]*(k+1) for _ in range(m)] for _ in range(n)]
  queue = deque()
  queue.append([0,0,0])
  visited[0][0][0] = 1

  while queue:
    x, y, z = queue.popleft()

    # 목표 지점에 도달하면 return
    if x==n-1 and y==m-1:
      return visited[x][y][z]-1

    # 상하좌우로 이동
    for (i,j) in dist:
      dx, dy = x+i, y+j
      if 0<=dx<n and 0<=dy<m and not visited[dx][dy][z] and not graph[dx][dy]:
        visited[dx][dy][z] = visited[x][y][z]+1
        queue.append([dx,dy,z])

    # 말 이동 수보다 작을 경우에만 이동
    if z<k:
      for (hi, hj) in horse:
        hx, hy = x+hi, y+hj
        if 0<=hx<n and 0<=hy<m:
          if not graph[hx][hy]:
            # z+1번째 말처럼 이동하는 중
            if not visited[hx][hy][z+1]:
              visited[hx][hy][z+1] = visited[x][y][z]+1
              queue.append([hx,hy,z+1])

  return -1

# 정답 출력
print(bfs())