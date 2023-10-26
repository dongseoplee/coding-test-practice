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