#15649번 N과 M (1)
import sys

n, m = map(int, sys.stdin.readline().split())
res = []


def bt():
    # 길이가 다 채워지면 출력
    if len(res) == m:
        print(' '.join(map(str, res)))
        return

    for j in range(1, n + 1):
        if j not in res:
            res.append(j)
            # print(res)
            # print('a')
            bt()
            res.pop()


bt()

#9663번 N-Queen
import sys
n = int(sys.stdin.readline())
v1 = [False for _ in range(n)]
v2 = [False for _ in range(2*n - 1)]
v3 = [False for _ in range(2*n - 1)]
global res
res = 0
def bt(cur):
  global res
  #탈출 조건
  if cur == n: #cur은 행 번호
    res += 1
    return

  for i in range(n):
    if v1[i] or v2[i+cur] or v3[cur-i+n-1]: #놓을수 없는 자리라면
      continue
    #놓을수있는 자리라면
    v1[i] = True
    v2[i+cur] = True
    v3[cur-i+n-1] = True
    bt(cur+1)
    v1[i] = False
    v2[i+cur] = False
    v3[cur-i+n-1] = False

bt(0)
print(res)

#1182번 부분수열의 합
import sys

n, s = map(int, sys.stdin.readline().split())
inputList = list(map(int, sys.stdin.readline().split()))
# print(n, s)

global res
res = 0


def bt(idx, sum):
    global res
    # 갯수가 n이면 탈출
    if idx == n:
        if sum == s:
            res += 1
        return

    bt(idx + 1, sum + 0)
    bt(idx + 1, sum + inputList[idx])
    # 트리 구조
    # 왼쪽 0 을 더하는 곳
    # 오른쪽 다음 수를 더하는 곳


bt(0, 0)

# 문제 조건 공집합 제외
if s == 0:
    print(res - 1)
else:
    print(res)

#15651번 N과 M (3)
import sys
n, m = map(int, sys.stdin.readline().split())
res = []
def bt(cnt):
  #탈출조건
  if cnt == m:
    print(*res)
    return
  for i in range(n):
    res.append(i+1)
    bt(cnt + 1)
    res.pop()

bt(0)

#15652번 N과 M (4)
import sys
n, m = map(int, sys.stdin.readline().split())
res = []
def bt(start):
  if len(res) == m:
    print(*res)
    return
  for i in range(start, n+1):
    res.append(i)
    bt(i)
    res.pop()

bt(1)

#15654번 N과 M (5)
import sys

n, m = map(int, sys.stdin.readline().split())
inputList = list(map(int, sys.stdin.readline().split()))
inputList = sorted(inputList)
# print(inputList)
res = []


def bt():
    if len(res) == m:
        print(*res)
        return

    for i in range(n):
        if inputList[i] not in res:
            res.append(inputList[i])
            bt()
            res.pop()


bt()

#15655번 N과 M (6)
import sys

n, m = map(int, sys.stdin.readline().split())
inputList = []
inputList = list(map(int, sys.stdin.readline().split()))
inputList = sorted(inputList)
# print(inputList)

resList = []


def bt(start):
    if len(resList) == m:
        print(*resList)
        return
    for i in range(start, len(inputList)):
        resList.append(inputList[i])
        bt(i + 1)
        resList.pop()


bt(0)

#15656번 N과 M (7)
import sys

n, m = map(int, sys.stdin.readline().split())
inputList = []
inputList = list(map(int, sys.stdin.readline().split()))
inputList = sorted(inputList)
# print(inputList)

resList = []


def bt():
    if len(resList) == m:
        print(*resList)
        return

    for i in range(len(inputList)):
        resList.append(inputList[i])
        bt()
        resList.pop()


bt()