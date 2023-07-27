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
