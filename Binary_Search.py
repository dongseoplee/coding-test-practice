#2110번 공유기 설치
import sys
n, c = map(int, sys.stdin.readline().split())
h = [int(sys.stdin.readline()) for i in range(n)]
h.sort()

start, end = 1, h[n-1] - h[0]
res = 0

if c == 2:
  print(h[n-1] - h[0])

else:
  while start < end:
    mid = (start + end) // 2
    cnt = 1
    ts = h[0]
    for i in range(n):
      if h[i]-ts >= mid:
        cnt += 1
        ts = h[i]

    if cnt >= c:
      res = mid
      start = mid + 1
    elif cnt < c:
      end = mid

  print(res)
