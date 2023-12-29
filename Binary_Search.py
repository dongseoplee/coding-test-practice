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

#3079번 입국심사
import sys
n, m = map(int, sys.stdin.readline().split()) #n개 심사대, m명
times = []
res = []
for _ in range(n):
  times.append(int(sys.stdin.readline()))

# print(times)

left = min(times)
right = max(times) * m

while left <= right:
  mid = (left + right) // 2
  total = 0
  for i in range(len(times)):
    total += mid//times[i]

  if total >= m:
    res.append(mid)
    right = mid - 1
  else:
    left = mid + 1

print(min(res))


#17266번 어두운 굴다리
import sys
n = int(sys.stdin.readline())
m = int(sys.stdin.readline())
position = list(map(int, sys.stdin.readline().split()))
len_positions = len(position)

min_height = 0

if len_positions == 1:
    min_height = max(position[0], n-position[0])
else:
    for i in range(len_positions):
        if i == 0:
            height = position[0]
        elif i == len_positions - 1:
            height = n - position[i]
        else:
            temp = position[i] - position[i-1]
            if temp % 2:
                height = temp // 2 + 1
            else:
                height = temp // 2

        min_height = max(height, min_height)
print(min_height)


