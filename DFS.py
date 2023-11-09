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

#2606번 바이러스
import sys

sys.setrecursionlimit(10 ** 9)

n = int(sys.stdin.readline())
graph = [[] for _ in range(n + 1)]
visited = [False] * (n + 1)
pairNum = int(sys.stdin.readline())
for _ in range(pairNum):
    a, b = map(int, sys.stdin.readline().split())
    graph[a].append(b)
    graph[b].append(a)

# print(graph)
cnt = 0


def dfs(graph, i, visited):
    global cnt
    cnt += 1
    # print("{}에 방문 함".format(i))
    visited[i] = True
    for a in graph[i]:
        if visited[a] == False:
            dfs(graph, a, visited)


dfs(graph, 1, visited)
print(cnt - 1)


#프로그래머스 타겟 넘버
def solution(numbers, target):
    # print(numbers, target)
    res = []

    def dfs(idx, sum_num):

        if idx == len(numbers):
            res.append(sum_num)
            return
        else:
            temp_idx = idx + 1
            temp_sum1 = sum_num + numbers[idx]
            dfs(temp_idx, temp_sum1)

            temp_sum2 = sum_num - numbers[idx]
            dfs(temp_idx, temp_sum2)

    dfs(0, 0)
    # print(cnt)
    # print(res)
    cnt = 0
    for resData in res:
        if resData == target:
            cnt += 1
    return cnt


#13565번 침투
import sys

sys.setrecursionlimit(3000000)


def dfs(y, x):
    graph[y][x] = 2
    for dy, dx in d:
        Y, X = y + dy, x + dx
        if (0 <= Y < M) and (0 <= X < N) and graph[Y][X] == 0:
            dfs(Y, X)


M, N = map(int, input().split())
graph = [list(map(int, list(input()))) for _ in range(M)]
d = [(-1, 0), (1, 0), (0, -1), (0, 1)]
for j in range(N):
    if graph[0][j] == 0:
        dfs(0, j)
print("YES" if 2 in graph[-1] else "NO")

#1987번 알파벳
import sys
r, c = map(int, sys.stdin.readline().split())
graph = []
for _ in range(r):
  temp = list(sys.stdin.readline().rstrip())
  graph.append(temp)

# visited = [[False for _ in range(c)] for _ in range(r)]
# print(graph)
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]
res = 0
def dfs(x, y, temp):
  global res
  res = max(res, temp)
  # print(x, y, temp)
  for i in range(4):
    nx = x + dx[i]
    ny = y + dy[i]
    if nx < 0 or nx >= r or ny < 0 or ny >= c:
      continue
    else:
      if graph[nx][ny] not in inputSet:
        inputSet.add(graph[nx][ny])
        dfs(nx, ny, temp + 1)
        #함수 끝나고 나와서 빼는 로직!! 추가해줘야함
        inputSet.remove(graph[nx][ny])

inputSet = set() #시간초과때문에 set 사용
inputSet.add(graph[0][0])
dfs(0, 0, 1)
print(res)

#1520번 내리막 길
# DFS + DP
import sys
input = sys.stdin.readline
sys.setrecursionlimit(10**6)

M, N = map(int, input().split())
dp = [[-1] * N for _ in range(M)]
arr = [list(map(int, input().split())) for _ in range(M)]
dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]

def dfs(x, y):
    # Base Case

    if x == M-1 and y == N-1:
        return 1
    # Visited Case
    if dp[x][y] != -1:
        return dp[x][y]
    dp[x][y] = 0
    for i in range(4):
        newx = x + dx[i]
        newy = y + dy[i]
        if 0 <= newx < M and 0 <= newy < N:
            if arr[newx][newy] < arr[x][y]:
                dp[x][y] += dfs(newx, newy)
    return dp[x][y]

print(dfs(0, 0))


#13549번 숨바꼭질 3
import heapq
INF = int(1e9)

N, K = map(int, input().split())  # 시작 위치, 도착 위치
distance = [INF]*100001  # 100001개의 떨어진 거리

def dijkstra(start):  # 다익스트라
    distance[start] = 0  # 시작 위치 초기화
    q = []
    heapq.heappush(q, (0, start))  # 시작 위치 우선 순위 큐 삽입

    while q:  # q에 값이 있을 동안
        dist, now = heapq.heappop(q)  # 거리가 가장 짧은 노드
        if distance[now] < dist:
            continue
        for a, b in [(now*2, dist), (now+1, dist+1), (now-1, dist+1)]:  # 2배, 오른쪽, 왼쪽
            if 0 <= a <= 100000 and distance[a] > b:  # 범위 안에 있고 방문하지 않았다면(범위 주의)
                distance[a] = b
                heapq.heappush(q, (b, a))

dijkstra(N)  # 시작 위치 다익스트라 실행
print(distance[K])  # 시작 위치로부터 K가 떨어진 최소 거리

#18352번 특정 거리의 도시 찾기
import sys
from collections import deque

queue = deque()
n, m, k, x = map(int, sys.stdin.readline().split())
graph = [[] for _ in range(n + 1)]
visited = [-1] * (n + 1)
for _ in range(m):
    a, b = map(int, sys.stdin.readline().split())
    graph[a].append(b)

queue.append(x)
visited[x] = 0
# bfs에서 너비 우선이므로 처음 방문이 최단(depth)으로 방문한 것이다. !!!!!!!
while queue:
    nowNode = queue.popleft()
    for nextNode in graph[nowNode]:
        if visited[nextNode] == -1:
            visited[nextNode] = visited[nowNode] + 1
            queue.append(nextNode)

if k in visited:
    for i in range(1, n + 1):
        if visited[i] == k:
            print(i)
else:
    print(-1)