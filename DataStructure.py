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

#2164번 카드2
import sys
from collections import deque
sys.stdin = open("input.txt", "r")
N = int(sys.stdin.readline())
q = deque([i for i in range(1, N+1)])

while True:
    if len(q) == 1:
        break
    q.popleft()
    num1 = q.popleft()
    q.append(num1)
print(q[0])

#1874번 스택 수열
import sys
from collections import deque
# sys.stdin = open("input.txt", "r")
n = int(sys.stdin.readline())
stack, ans, find = [], [], True
now = 1
for _ in range(n):
    num = int(sys.stdin.readline())
    while now <= num:
        stack.append(now)
        ans.append('+')
        now += 1
    if stack[-1] == num:
        stack.pop()
        ans.append('-')
    else:
        find = False

if not find:
    print('NO')
else:
    for i in ans:
        print(i)
