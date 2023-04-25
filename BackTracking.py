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