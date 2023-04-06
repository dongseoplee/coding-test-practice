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