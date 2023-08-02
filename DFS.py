#11724번 연결 요소의 개수
import sys  # dfs + 인접리스트 문제

sys.setrecursionlimit(10000)


def dfs(v):
    visited[v] = True #노드를 방문처리 하고
    for k in graph[v]: #연결된 노드를 꺼내서
        if visited[k] == False: #방문하지 않았다면 dfs 함수 호출
            dfs(k)


n, m = map(int, sys.stdin.readline().split())
graph = [[] for _ in range(n + 1)]
visited = [False] * (n + 1)
cnt = 0
for _ in range(m):
    u, v = map(int, sys.stdin.readline().split())
    graph[u].append(v)
    graph[v].append(u)

for j in range(1, n + 1):
    if visited[j] == False:  # 방문하지 않았다면
        dfs(j)
        cnt += 1

print(cnt)

#11725번 트리의 부모 찾기
import sys

sys.setrecursionlimit(10 ** 6)
n = int(sys.stdin.readline())  # dfs는 항상 부모에서 자식으로 이동한다.
graph = [[] for _ in range(n + 1)]
visited = [False] * (n + 1)
for _ in range(n - 1):
    a, b = map(int, sys.stdin.readline().split())
    graph[a].append(b)
    graph[b].append(a)


def dfs(i):
    for node in graph[i]:
        if visited[node] == False:  # 방문하지 않았다면
            visited[node] = i
            dfs(node)


dfs(1)  # dfs는 항상 부모에서 자식으로 이동한다.
for k in range(2, n + 1):
    print(visited[k])

#2644번 촌수 계산
import sys
sys.setrecursionlimit(10**6)
n = int(sys.stdin.readline())
a, b = map(int, sys.stdin.readline().split())
m = int(sys.stdin.readline())
graph = [[] for _ in range(n+1)]
visited = [False] * (n+1)
for _ in range(m):
  x, y = map(int, sys.stdin.readline().split())
  graph[x].append(y)
  graph[y].append(x)

# print(graph)

def dfs(i, cnt): #촌수 계산을 위해 dfs 함수 실행할때마다 cnt + 1 해줌
  cnt += 1
  for node in graph[i]:
    if visited[node] == False:
      visited[node] = cnt
      dfs(node, cnt)

dfs(a, 0)
# print(visited)

if visited[b] == False:
  print(-1)
elif visited[b] != False:
  print(visited[b])

#1325번 효율적인 해킹
import sys

sys.setrecursionlimit(10 ** 6)
n, m = map(int, sys.stdin.readline().split())
graph = [[] for _ in range(n + 1)]
visited = [False for _ in range(n + 1)]
res = [0 for _ in range(n + 1)]
for _ in range(m):
    a, b = map(int, sys.stdin.readline().split())
    graph[b].append(a)


# print(graph)
# print(visited)

def dfs(i, cnt):
    cnt += 1
    for node in graph[i]:
        if visited[node] == False:
            visited[node] = cnt
            dfs(node, cnt)


for i in range(1, n + 1):
    dfs(i, 1)
    res[i] = max(visited)
    visited = [False for _ in range(n + 1)]

for j in range(1, n + 1):
    if res[j] == max(res):
        print(j, end=' ')

#2210번 숫자판 점프
import sys  # 2차원 그래프에서 dfs 적용법

graph = []
result = []
for _ in range(5):
    graph.append(list(map(str, sys.stdin.readline().split())))  # map에서 int가 아닌 str로 선언


def dfs(x, y, number):
    if len(number) == 6:
        if number not in result:
            result.append(number)
        return
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    for k in range(4):
        nx = x + dx[k]
        ny = y + dy[k]
        if nx < 0 or nx >= 5 or ny < 0 or ny >= 5:
            continue
        else:
            dfs(nx, ny, number + graph[nx][ny])


for i in range(5):
    for j in range(5):
        dfs(i, j, graph[i][j])
    # print(graph)
print(len(result))

#13023번 ABCDE
import sys

sys.setrecursionlimit(10 ** 6)
n, m = map(int, sys.stdin.readline().split())
graph = [[] for _ in range(n)]
visited = [False] * n
for _ in range(m):
    a, b = map(int, sys.stdin.readline().split())
    graph[a].append(b)
    graph[b].append(a)


# print(graph)

def dfs(i, cnt):
    visited[i] = True  # 첫번째 시작 노드 방문처리
    cnt += 1
    if cnt == 5:
        print(1)
        exit()
    for node in graph[i]:
        if visited[node] == False:
            dfs(node, cnt)
            visited[node] = False  # dfs에서 빠져나온 상황은 제일 안쪽까지 갔다가 나온것이므로 방문처리 풀어줌!!


for i in range(n):
    dfs(i, 0)  # 모든 노드에 대해서 수행해보고
    visited = [False] * n  # visited 초기화 해주고

print(0)

#1260번
import sys
from collections import deque

sys.setrecursionlimit(10 ** 6)
n, m, v = map(int, sys.stdin.readline().split())
graph = [[] for _ in range(n + 1)]
visited = [False] * (n + 1)
for _ in range(m):
    a, b = map(int, sys.stdin.readline().split())
    graph[a].append(b)
    graph[b].append(a)

# print(graph)
for k in range(1, n + 1):  # 정점 번호가 작은 것 부터 방문하도록 오름차순 정렬시킴
    graph[k].sort()


def dfs(i):
    visited[i] = True
    print(i, end=' ')
    for node in graph[i]:
        if visited[node] == False:
            dfs(node)


dfs(v)
print()
queue = deque()
visited = [False] * (n + 1)


def bfs(i):
    print(i, end=' ')
    visited[i] = True
    for node in graph[i]:
        queue.append(node)
        visited[node] = True

    while queue:
        popNum = queue.popleft()
        print(popNum, end=' ')
        for a in graph[popNum]:
            if visited[a] == False:
                queue.append(a)
                visited[a] = True


bfs(v)