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
