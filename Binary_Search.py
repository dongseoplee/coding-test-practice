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

#1920 수 찾기
import sys

#이진 탐색의 기본은 오름차순 정렬을 해두는 것
n = int(sys.stdin.readline())
nl = list(map(int, sys.stdin.readline().split()))
m = int(sys.stdin.readline())
ml = list(map(int, sys.stdin.readline().split()))
nl.sort() # 오름차순 정렬 메소드 sort()

def binary_search(arr, start, end, target):
  while start <= end:
    mid = int((start + end) // 2)
    if arr[mid] < target:
      start = mid + 1
    elif arr[mid] > target:
      end = mid - 1
    elif arr[mid] == target:
      return 1

  return 0


for i in range(m):
  print(binary_search(nl, 0, n-1, ml[i]))

#1822번 차집합
import sys

na, nb = map(int, sys.stdin.readline().split())
a = list(map(int, sys.stdin.readline().split()))
b = list(map(int, sys.stdin.readline().split()))
# print(a)
b.sort()
# print(b)

def binary_search(arr, target, start, end):
  while start <= end:
    mid = (start + end) // 2
    if arr[mid] < target:
      start = mid + 1
    elif arr[mid] > target:
      end = mid - 1
    elif arr[mid] == target:
      return True
  return False
res = []
for i in range(na):
  if binary_search(b, a[i], 0, nb-1) == False: #원소가 b에도 없다면
    res.append(a[i])


if len(res) == 0:
  print(0)
else:
  print(len(res))
  res.sort()
  print(*res)