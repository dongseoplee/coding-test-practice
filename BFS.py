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


