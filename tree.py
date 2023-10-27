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
# 7
# A B C
# B D .
# C E F
# E . .
# F . G
# D . .
# G . .