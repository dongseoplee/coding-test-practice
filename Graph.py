#1753번 최단경로 (다익스트라)
import sys, heapq

input = sys.stdin.readline
INF = int(1e9)

V, E = map(int, input().split())
K = int(input())
graph = [[] for _ in range(V + 1)]
for _ in range(E):
    u, v, w = map(int, input().split())
    graph[u].append((w, v))
이
dist = [INF] * (V + 1)
q = []


def Dijkstra(start):
    dist[start] = 0
    heapq.heappush(q, (0, start))

    while q:
        current_weight, current_node = heapq.heappop(q)

        if dist[current_node] < current_weight: continue

        for next_weight, next_node in graph[current_node]:
            distance = next_weight + current_weight
            if distance < dist[next_node]:
                dist[next_node] = distance
                heapq.heappush(q, (distance, next_node))


Dijkstra(K)
for i in range(1, V + 1):
    print("INF" if dist[i] == INF else dist[i])