#2741번 N 찍기
import sys

n = int(sys.stdin.readline())
for i in range(1, n+1):
  print(i)

#2742번 기찍 N
import sys

n = int(sys.stdin.readline())
for i in range(n, 0, -1):
  print(i)

#1110번 더하기 사이클
import sys

n = int(sys.stdin.readline())
newNum = -1
cnt = 0
temp = n
while(1):
  if n == newNum:
    break
  newNum = (temp//10 + temp%10)%10 + (temp%10)*10
  cnt += 1
  temp = newNum

print(cnt)

#2577번 숫자의 개수
import sys
a = int(sys.stdin.readline())
b = int(sys.stdin.readline())
c = int(sys.stdin.readline())
count = [0]*10
num = a*b*c
while num != 0:
  count[num%10] += 1
  num = num // 10

# print(count)
for i in range(len(count)):
  print(count[i])


#1913번 달팽이
import sys

n = int(sys.stdin.readline())
num = int(sys.stdin.readline())
graph = [[0 for _ in range(n)] for _ in range(n)]
direction = [(1, 0), (0, 1), (-1, 0), (0, -1)]
directionChange = 0
nx = -1
ny = 0
for i in range(n ** 2, 0, -1):
  mx, my = direction[directionChange % 4]
  nx += mx
  ny += my  # 다음 좌표 확인하고
  if nx < 0 or nx >= n or ny < 0 or ny >= n or graph[nx][ny] != 0:  # 인덱스 범위 넘어감
    nx -= mx
    ny -= my  # 원래대로 돌리고
    directionChange += 1
    mx, my = direction[directionChange % 4]
    nx += mx
    ny += my

  graph[nx][ny] = i

for graphRow in graph:
  print(*graphRow)

for j in range(n):
  for k in range(n):
    if graph[j][k] == num:
      print(j + 1, k + 1)

#18406번 럭키 스트레이트
import sys

n = int(sys.stdin.readline())
num1 = []
while n > 0:
  num1.append(int(n % 10))
  n = n // 10

# print(num1)
sum1 = 0
for i in range(len(num1) // 2):
  sum1 += num1[i]

# listSum = sum(num1)
# print(listSum)

if sum1 * 2 == int(sum(num1)):
  print('LUCKY')
else:
  print('READY')