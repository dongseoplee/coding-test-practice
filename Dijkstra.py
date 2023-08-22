#1916번 최소비용 구하기
import sys
import heapq

n = int(sys.stdin.readline())
m = int(sys.stdin.readline())
graph = [[] for _ in range(n+1)]
for _ in range(m):
  a, b, c = map(int, sys.stdin.readline().split())
  graph[a].append((b, c))

start, end = map(int, sys.stdin.readline().split())
INF = 1e9
distance = [INF]*(n+1)
def dijkstra(start):
  q = []
  heapq.heappush(q, (0, start))
  distance[start] = 0
  while q:
    dist, now = heapq.heappop(q) # 최소 힙 (0, 1)
    if distance[now] < dist:
      continue

    for i in graph[now]:
      cost = dist + i[1]
      if cost < distance[i[0]]:
        distance[i[0]] = cost
        heapq.heappush(q, (cost, i[0]))

dijkstra(start)
print(distance[end])

#11779번 최소비용 구하기 2
import sys
import heapq

n = int(sys.stdin.readline())
m = int(sys.stdin.readline())
graph = [[] for _ in range(n + 1)]
route = [[] for _ in range(n + 1)]
for _ in range(m):
  a, b, c = map(int, sys.stdin.readline().split())
  graph[a].append((b, c))  # 비용, 도착 노드

start, end = map(int, sys.stdin.readline().split())

INF = int(1e9)
nearest = [start] * (n + 1)
distance = [INF] * (n + 1)
q = [(0, start)]
while q:
  dist, now = heapq.heappop(q)
  if dist > distance[now]:
    continue

  for next, nextDist in graph[now]:
    cost = nextDist + dist
    if cost < distance[next]:
      distance[next], nearest[next] = cost, now
      heapq.heappush(q, (cost, next))

res = []
temp = end
while temp != start:
  res.append(str(temp))
  temp = nearest[temp]

res.append(str(start))
res.reverse()

print(distance[end])
print(len(res))
print(" ".join(res))
# print(distance[end])
# print(nearest)