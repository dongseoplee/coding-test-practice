#10773번 제로
import sys
from collections import deque
# sys.stdin = open("input.txt", "r")
q = deque()
K = int(sys.stdin.readline())
for _ in range(K):
    inum = int(sys.stdin.readline())
    if inum == 0:
        q.pop()
    else:
        q.append(inum)
print(sum(q))

#17298번 오큰수
import sys
from collections import deque
sys.stdin = open("input.txt", "r")
N = int(sys.stdin.readline())
A = list(map(int, sys.stdin.readline().split()))
ans = [-1] * (N)
q = deque()
for i in range(N):
    while q and q[-1][0] < A[i]:
        inum, idx = q.pop()
        ans[idx] = A[i]
    q.append((A[i], i))             #값과 인덱스 저장.
print(*ans)