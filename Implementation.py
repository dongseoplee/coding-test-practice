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

#10431번 줄세우기
import sys

testNum = int(sys.stdin.readline())
graph = []
for _ in range(testNum):
  graph.append(list(map(int, sys.stdin.readline().split())))
# print(graph)
height = [[] for _ in range(testNum)]
res = [[] for _ in range(testNum)]

cnt = 0
for i in range(testNum):
  cnt = 0
  height[i].append(graph[i][0])
  height[i].append(graph[i][1])
  for j in range(2, 21):
    for k in range(1, len(height[i])):
      if height[i][k] > graph[i][j]:
        height[i].insert(k, graph[i][j])
        cnt += len(height[i]) - 1 - k
        break #가장 가까운 for문 하나만 break
      if k == len(height[i]) - 1:
        height[i].append(graph[i][j])
  res[i].append(graph[i][0])
  res[i].append(cnt)

# print(res)
for resData in res:
  print(resData[0], resData[1])


#14719번 빗물
import sys
h, w = map(int, sys.stdin.readline().split())
graph = [[False for _ in range(w)] for _ in range(h)]
water = list(map(int, sys.stdin.readline().split()))
# print(water)
for i in range(len(water)):
  waterSize = water[i]
  for j in range(waterSize):
    graph[j][i] = True

# print(graph)
res = 0
for k in range(h): #4
  temp = []
  for l in range(w): #8
    # print(k, l)
    if graph[k][l] == True:
      # print(l)
      temp.append(l)
  # print(temp)

  if len(temp) > 1:
    # print(temp)
    for m in range(1, len(temp)):
      res += temp[m] - temp[m-1] - 1
      # print("res", res)
print(res)

#프로그래머스 주차 요금 계산
import math
def solution(fees, records):
    answer = []
    cars = set()
    for record in records:
        car_time, car_num, car_state = record.split(' ')
        # print(car_time, car_num, car_state)
        cars.add(car_num)
    carsList = list(cars)
    carsList.sort()
    res = [[] for _ in range(len(carsList))]
    idx = 0
    for car in carsList:
        res[idx].append(car)
        for record in records:
            car_time, car_num, car_state = record.split(' ')
            if int(car) == int(car_num):
                hh, mm = car_time.split(':')
                res[idx].append(int(hh)*60 + int(mm))
        idx += 1
    # print(res)
    for i in range(len(res)):
        if len(res[i]) % 2 == 0: #짝수면 23시59분 넣어줘라
            res[i].append(23*60 + 59)
    print(res)
    #총시간 구하기
    timeSum = []
    for j in range(len(res)):
        sum = 0
        for k in range(1, len(res[j]), 2):
            print("k", k)
            print(res[j][k+1]-res[j][k])
            sum += res[j][k+1]-res[j][k]
        timeSum.append(sum)
    print(timeSum)
    # 요금계산하기
    for i in range(len(timeSum)):
        if timeSum[i] <= fees[0]:
            answer.append(fees[1])
        else:
            answer.append(math.ceil((timeSum[i] - fees[0]) / fees[2])*fees[3] + fees[1])
    return answer

#11723번 집합
import sys
m = int(sys.stdin.readline())
# print(m)
s = set()
for _ in range(m):
    command = sys.stdin.readline().rstrip()
    commandList = command.split()
    # print(commandList)
    if commandList[0] == 'add':
        if int(commandList[1]) not in s:
            s.add(int(commandList[1]))
    if commandList[0] == 'remove':
        if int(commandList[1]) in s:
            s.remove(int(commandList[1]))
    if commandList[0] == 'check':
        if int(commandList[1]) in s:
            print(1)
        else:
            print(0)
    if commandList[0] == 'toggle':
        if int(commandList[1]) in s:
            s.remove(int(commandList[1]))
        else:
            s.add(int(commandList[1]))
    if commandList[0] == 'all':
        s = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
    if commandList[0] == 'empty':
        s = set()

#8979번 올림픽
import sys
n, k = map(int, sys.stdin.readline().split())
country = []
for _ in range(n):
    country.append(list(map(int, sys.stdin.readline().split())))

country.sort(key=lambda x: (-x[1], -x[2], -x[3]))
idx = 0
for i in range(n):
    if country[i][0] == k:
        idx = i

for i in range(n):
    if country[idx][1:] == country[i][1:]:
        print(i+1) #등수가 같으면 첫번째 동순위 idx에서 끝난다.
        break

