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