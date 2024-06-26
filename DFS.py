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

#9466번 텀 프로젝트
import sys

sys.setrecursionlimit(10 ** 6)
input = sys.stdin.readline


def dfs(x, result):
    visited[x] = True
    cycle.append(x)  # 사이클을 이루는 팀을 확인하기 위함
    number = arr[x]

    if visited[number]:  # 방문가능한 곳이 끝났는지
        if number in cycle:  # 사이클 가능 여부
            result += cycle[cycle.index(number):]  # 사이클 되는 구간 부터만 팀을 이룸 사이클이 있으면 해당 인덱스에서 잘라서 res 리스트에 이어 붙이기
        return
    else:
        dfs(number, result)


for _ in range(int(input())):
    n = int(input())
    arr = [0] + list(map(int, input().split()))
    visited = [False for _ in range(n + 1)]
    result = []

    for i in range(1, n + 1):
        if not visited[i]:  # 방문 안한 곳이라면
            cycle = []
            dfs(i, result)  # DFS 함수 돌림

    print(n - len(result))  # 팀에 없는 사람 수
    print(result)

# 내 풀이
import sys
from collections import deque
testNum = int(sys.stdin.readline())
for _ in range(testNum):
    n = int(sys.stdin.readline())
    graph = [0]
    graph += list(map(int, sys.stdin.readline().split()))
    # print(graph)
    # print(visited)
    visitedAll = [False] *(n+1)
    queue = deque()

    def dfs(i):
        temp = []
        queue.append(i)
        temp.append(i)
        visited = [False] * (n + 1)
        visited[i] = True
        while queue:
            x = queue.popleft()
            # print("x", x)
            if visited[graph[x]] == False:
                queue.append(graph[x])
                temp.append(graph[x])
                visited[x] = True
            if visited[graph[x]] == True and graph[x] == i:
                # print("팀 형성")
                #visited True 인 것 갯수 세기
                for tempData in temp:
                    visitedAll[tempData] = True
                # for j in range(1, n+1): # 이 지점 for문 시간 오래 걸릴 것으로 예상
                #     if visited[j] == True:
                #         visitedAll[j] = True


    for i in range(1, n+1):
        if visitedAll[i] == False:
            dfs(i)

    print(visitedAll.count(False) - 1)

#1707번 이분 그래프
import sys
sys.setrecursionlimit(int(1e9))
k = int(sys.stdin.readline())

def dfs(now, group):
    #이분 그래프: 인접 노드와 다른 색상 칠한다.
    visited[now] = group
    for nextNode in graph[now]:
        if visited[nextNode] == False:
            if dfs(nextNode, group*(-1)) == False:
                return False
        else: # 방문 한곳
            if visited[nextNode] == visited[now]:
                return False
    return True
for _ in range(k):
    v, e = map(int, sys.stdin.readline().split())
    graph = [[] for _ in range(v+1)]
    visited = [False for _ in range(v+1)]
    for _ in range(e):
        u, v = map(int, sys.stdin.readline().split())
        graph[u].append(v)
        graph[v].append(u)
    # print(graph)
    res = True
    for i in range(1, v+1):
        if visited[i] == False:
            res = dfs(i, 1)
            if res == False:
                break

    if res:
        print("YES")
    else:
        print("NO")


#10451번 순열 사이클
import sys
sys.setrecursionlimit(2000) #dfs 항상 설정 해주기
testNum = int(sys.stdin.readline())

def dfs(start):
    # print(start)
    visited[start] = True #방문 처리
    if visited[nodes[start]] == False:
        dfs(nodes[start])
    elif visited[nodes[start]] == True:
        return


for _ in range(testNum):

    n = int(sys.stdin.readline())
    nodes = [0] + list(map(int, sys.stdin.readline().split()))
    visited = [False for _ in range(n+1)]
    res = 0
    for i in range(1, n+1):
        if visited[i] == False:
            res += 1
            dfs(i)
    print(res)

#1167번 트리의 지름
import sys #한 지점에서 가장 먼 곳을 찾고 그 곳에서 가장 먼곳을 찾으면 지름이 구해진다.
sys.setrecursionlimit(10**9)
v = int(sys.stdin.readline())
graph = [[] for _ in range(v+1)]
visited = [False for _ in range(v+1)]

for _ in range(v):
    temp = list(map(int, sys.stdin.readline().split()))
    for j in range(1, len(temp)-2, 2): #2씩 간격을 두고 graph 만들기
        graph[temp[0]].append([temp[j], temp[j+1]])

def dfs(x, y):
    for node, dis in graph[x]:
        if visited[node] == False: #방문 안한노드
            visited[node] = dis + y
            dfs(node, dis+y)

visited[1] = True
dfs(1, 0)
# print(visited)
startNode = visited.index(max(visited))
visited = [False for _ in range(v+1)]
visited[startNode] = True
dfs(startNode, 0)
print(max(visited))

#1068번 트리
import sys
n = int(sys.stdin.readline())
nodes = [[] for _ in range(n)] # 자식 노드만 담는 배열
a = list(map(int, sys.stdin.readline().split()))
for i in range(n):
    if a[i] == -1:
        continue
    else:
        nodes[a[i]].append(i)
deleteNum = int(sys.stdin.readline())

visited = [False for _ in range(n)]

def dfs(node):
    visited[node] = True
    for x in nodes[node]:
        if visited[x] == False:
            dfs(x)

dfs(deleteNum)

res = 0
for i in range(n):
    if visited[i] == False and not nodes[i]:
        res += 1
    if deleteNum in nodes[i] and len(nodes[i]) == 1: #삭제되는 노드의 부모 노드가 자식이 삭제되는 노드 1개 였는지 확인
        res += 1

print(res)

#1937번 욕심쟁이 판다
import sys
sys.setrecursionlimit(10**9)
n = int(sys.stdin.readline())
graph = []
for _ in range(n):
    graph.append(list(map(int, sys.stdin.readline().split())))
# print(graph)
dp = [[0] * n for _ in range(n)]
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

def dfs(x, y):
    if dp[x][y]:
        return dp[x][y]
    dp[x][y] = 1
    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
        if nx < 0 or nx >= n or ny < 0 or ny >= n:
            continue
        if graph[x][y] < graph[nx][ny]:
            dp[x][y] = max(dp[x][y], dfs(nx, ny) + 1)
    return dp[x][y]

for i in range(n):
    for j in range(n):
        if not dp[i][j]:
            dfs(i, j)
res = 0
for m in range(n):
    res = max(res, max(dp[m]))

print(res)

#2638번 치즈
import sys #치즈 내부 공간인지 어떻게 알지? 맨 가장자리는 치즈가 놓이지 않으므로 0,0에서 DFS로 바깥 공기 확인 가능
sys.setrecursionlimit(10**6)
n, m = map(int, sys.stdin.readline().split())
visited = [[False for _ in range(m)] for _ in range(n)]
graph = []
for _ in range(n):
    graph.append(list(map(int, sys.stdin.readline().split())))

# print(graph)
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]
def outer(x, y, visited):
    graph[x][y] = 2
    visited[x][y] = True
    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
        if nx < 0 or nx >= n or ny < 0 or ny >= m:
            continue
        elif visited[nx][ny] == False and (graph[nx][ny] == 0 or graph[nx][ny] == 2):
            visited[nx][ny] = True
            graph[nx][ny] = 2
            outer(nx, ny, visited)


def countOutNum(a, b):
    cnt = 0
    for i in range(4):
        nx = a + dx[i]
        ny = b + dy[i]
        if 0<=nx<n and 0<=ny<m:
            if graph[nx][ny] == 2:
                cnt += 1
    return cnt
res = 0
while True:
    res += 1
    visited = [[False for _ in range(m)] for _ in range(n)]
    outer(0, 0, visited)
    rmList = []
    for i in range(n):
        for j in range(m):
            if graph[i][j] == 1 and countOutNum(i, j) >= 2:
                rmList.append((i, j))


    for x, y in rmList:
        graph[x][y] = 2
    flag = True
    for i in range(n):
        for j in range(m):
            if graph[i][j] == 1:
                flag = False
    if flag == True:
        break
print(res)


#음료수 얼려 먹기
import sys
sys.setrecursionlimit(10**9)
n, m = map(int, sys.stdin.readline().split())
graph = []
visited = [[False for _ in range(m)] for _ in range(n)]
for _ in range(n):
    graph.append(list(sys.stdin.readline().rstrip()))

print(graph)
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]
#방문처리 목적 dfs
def dfs(x, y):
    visited[x][y] = True
    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
        if nx<0 or nx >=n or ny < 0 or ny >=m:
            continue
        if graph[nx][ny] == '0' and visited[nx][ny] == False:
            dfs(nx, ny)


res = 0
for i in range(n):
    for j in range(m):
        if graph[i][j] == '0' and visited[i][j] == False:
            dfs(i, j)
            res += 1

print(res)