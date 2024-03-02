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

#15657번 N과 M (8)
import sys

n, m = map(int, sys.stdin.readline().split())
# n, m = 4, 2
arr = sorted(list(map(int, input().split())))

temp = []
def bt(start):
  if len(temp) == m:
    # temp.append(res)
    print(*temp)
    return

  for i in range(start, n):
    temp.append(arr[i])
    bt(i)
    temp.pop()

bt(0)

#15650 N과 M(2)
import sys
# sys.stdin = open("input.txt", "r")

def dfs(n, s, lst): #n 나타낼 수 들
    # 1. 종료조건 처리(n에 관련되게)!!! + 정답처리
    if n == M:
        ans.append(lst)
        return
    # 2. 하부 함수 호출
    for j in range(s, N+1):
        dfs(n+1, j+1, lst+[j])
    # dfs(n+1, lst+[n]) #숫자 선택하는 경우
    # dfs(n+1, lst) #숫자 선택하지 않는 경우

N, M = map(int, sys.stdin.readline().split())
ans = [] # 정답 저장 리스트
v = [0] * (N+1) # 중복 확인

dfs(0, 1, [])
for lst in ans:
    print(*lst)

#15651 N과 M(3)
import sys
sys.stdin = open("input.txt", "r")

def dfs(n, lst): #n 선택한 숫자 갯수
    # 1. 종료조건 처리(n에 관련되게)!!! + 정답처리
    if n == M:
        ans.append(lst)
        return
    # 2. 하부 함수 호출
    for j in range(1, N+1):
        dfs(n+1, lst+[j])

N, M = map(int, sys.stdin.readline().split())
ans = [] # 정답 저장 리스트

dfs(0, [])
for lst in ans:
    print(*lst)