#9934번 완전 이진 트리
import sys
k = int(sys.stdin.readline())
nums = list(map(int, sys.stdin.readline().split()))
idxList = [2, 1, 2]
res = [[] for _ in range(k+1)]
for i in range(1, k-1):
    temp = []
    for j in range(len(idxList)):
        temp.append(idxList[j] + 1)
    idxList = temp + [1] + temp

# print(nums)
# print(idxList)
for i in range(1, k+1): #층 수
    for j in range(len(idxList)):
        if idxList[j] == i:
            res[i].append(nums[j])

for i in range(k):
    print(*res[i+1])
# print(res)

#1991번 트리 순회
# 전위 순회(preorder traverse) : 뿌리(root)를 먼저 방문
# 중위 순회(inorder traverse) : 왼쪽 하위 트리를 방문 후 뿌리(root)를 방문
# 후위 순회(postorder traverse) : 하위 트리 모두 방문 후 뿌리(root)를 방문
# 전위 순회한 결과 : ABDCEFG // (루트) (왼쪽 자식) (오른쪽 자식)
# 중위 순회한 결과 : DBAECFG // (왼쪽 자식) (루트) (오른쪽 자식)
# 후위 순회한 결과 : DBEGFCA // (왼쪽 자식) (오른쪽 자식) (루트)
# 트리를 어떻게 만드는 지가 궁금
import sys
n = int(sys.stdin.readline())
tree = {}
for _ in range(n):
    root, left, right = sys.stdin.readline().split()
    tree[root] = [left, right]

# print(tree)

def preorder(root):
    if root != '.':
        print(root, end='')
        preorder(tree[root][0])
        preorder(tree[root][1])

def inorder(root):
    if root != '.':
        inorder(tree[root][0])
        print(root, end='')
        inorder(tree[root][1])

def postorder(root):
    if root != '.':
        postorder(tree[root][0])
        postorder(tree[root][1])
        print(root, end='')


# def postorder(root):
#
preorder('A')
print()
inorder('A')
print()
postorder('A')

#5639번 이진 검색 트리
import sys
sys.setrecursionlimit(10**9)
nums = []
while True:
    try:
        nums.append(int(sys.stdin.readline()))
    except:
        break

# print(nums)

def postorder(start, end):
    if start > end:
        return
    mid = end + 1
    for i in range(start + 1, end + 1): #1~8
        if nums[i] > nums[start]: # 루트보다 큰 값까지
            mid = i
            break
    postorder(start+1, mid-1)
    postorder(mid, end)
    print(nums[start])

postorder(0, len(nums) - 1)

#1967번 트리의 지름
import sys #임의의 한점에서 가장먼 거리 노드 A 찾고 A 에서 가장 먼 거리 노드  B 찾기 #임의로 아무 노드로 시작하는 점이 중요!
from collections import deque
n = int(sys.stdin.readline())
tree = [[] for _ in range(n+1)]
tree2 = [[] for _ in range(n+1)]
# print(tree)
for _ in range(n-1):
    a, b, c = map(int, sys.stdin.readline().split())
    tree[a].append([b, c])
    tree2[a].append([b, c])
    tree2[b].append([a, c])
    # tree[b].append([a, c])

queue = deque()
visited = [False] * (n+1)
def bfs():
    maxDist = 0
    maxNode = -1
    while queue:
        nowNode, nowNodeDist = queue.popleft()
        if nowNodeDist >= maxDist:
            maxNode = nowNode
            maxDist = nowNodeDist
        # print(nowNode, nowNodeDist)
        # print()
        # maxDist = max(maxDist, nowNodeDist)
        for nextNode, nextNodeDist in tree[nowNode]:
            # print(nextNode, nextNodeDist)
            queue.append([nextNode, nextNodeDist + nowNodeDist])
            # print(nextNode, nextNodeDist + nowNodeDist)
            # if visited[nextNode] == False:
            #     queue.append([nextNode, nextNodeDist + nowNodeDist])
            #     visited[nextNode] = True
    return maxNode, maxDist





queue.append([1, 0]) # 1을 시작 노드로
tempNode, tempDist = bfs()
queue.append([tempNode, 0])
visited[tempNode] = True

def bfs2():
    # print("bfs2")
    # print(queue)
    maxNode = -1
    maxDist = 0
    while queue:
        nowNode, nowDist = queue.popleft()
        if nowDist >= maxDist:
            maxNode = nowNode
            maxDist = nowDist
        # print(nowNode, nowDist)
        # print(visited[nowNode])
        for nextNode, nextDist in tree2[nowNode]:
            if visited[nextNode] == False:
                queue.append([nextNode, nowDist + nextDist])
                # print(nextNode, nowDist + nextDist)
                visited[nextNode] = True
                # print(nextNode, nowDist + nextDist)
    return maxDist
# print(bfs2())

print(bfs2())
# print(queue.popleft())