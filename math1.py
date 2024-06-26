#2745번 진법 변환
import sys
n, b = sys.stdin.readline().split()
# print(n, b)
b = int(b)
res = 0

numDict = dict()
for j in range(48, 58):
  numDict[chr(j)] = j - 48
for k in range(65, 91):
  numDict[chr(k)] = k - 55
#48~57 숫자 0부터9
#65부터90까지 A부터Z
# print(numDict)

# print(type(numDict['0']))
for i in range(len(n)):
  res += numDict[n[i]] * (b**(len(n)-i-1))
print(res)

#11005번 진법 변환 2
import sys
n, b = map(int, sys.stdin.readline().split())
modList = []
while n != 0:
  modList.append(n%b)
  n = n // b

numDict = dict()
for j in range(48, 58):
  numDict[j - 48] = chr(j)
for k in range(65, 91):
  numDict[k - 55] = chr(k)

# print(numDict)
# print(modList)
#idx 큰것부터 출력해야함
for i in range(len(modList)):
  print(numDict[modList[len(modList) - i - 1]], end='')

#2720번 세탁소 사장 동혁
import sys
testNum = int(sys.stdin.readline())
for _ in range(testNum):
  res = []
  price = int(sys.stdin.readline())
  res.append(price // 25)
  price = price % 25
  res.append(price // 10)
  price = price % 10
  res.append(price // 5)
  price = price % 5
  res.append(price)
  print(*res)


#2903번 중앙 이동 알고리즘
import sys
import math
n = int(sys.stdin.readline())
dotList = []
dotList.append(4)
cnt = 0
idx = 0

while True:
  if cnt == n:
    break
  res = (math.sqrt(dotList[len(dotList)-1]) + (2**idx))**2
  dotList.append(res)
  idx += 1
  cnt += 1

print(int(dotList[len(dotList) - 1]))

#10757번 큰 수 A + B
import sys
a, b = map(int, sys.stdin.readline().split())
print(a+b)

#2292번 벌집
import sys

n = int(sys.stdin.readline())
nums, cnt = 1, 1
while n > nums:
  nums += 6 * cnt
  cnt += 1

print(cnt)

#2869번 달팽이는 올라가고 싶다.
import sys
import math
a, b, v = map(int, sys.stdin.readline().split())

day = (v-b) / (a-b)
print(math.ceil(day))

#1193번 분수찾기
import sys

n = int(sys.stdin.readline())

line = 0
maxNum = 0
while n > maxNum:
  line += 1
  maxNum += line

# print(line)
# print(maxNum)
# 짝수 라인
if line % 2 == 0:
  bj = line - (maxNum - n)
  bm = maxNum - n + 1
# 홀수 라인
else:
  bj = maxNum - n + 1
  bm = line - (maxNum - n)

print(bj, end='')
print('/', end='')
print(bm)


#23971번 ZOAC 4
import sys
import math
h, w, n, m = map(int, sys.stdin.readline().split())
res = math.ceil(h/(n+1)) * math.ceil(w/(m+1))
print(res)