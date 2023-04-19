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




