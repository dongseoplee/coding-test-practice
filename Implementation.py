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

#16236번 아기 상어
import sys
from collections import deque
n = int(sys.stdin.readline())
INF = 1e9
graph = []
for _ in range(n):
    graph.append(list(map(int, sys.stdin.readline().split())))

# print(graph)
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]
shark_size = 2
nowX, nowY = 0, 0

for i in range(n):
    for j in range(n):
        if graph[i][j] == 9:
            nowX, nowY = i, j
            graph[nowX][nowY] = 0

def bfs():
    visited = [[-1 for _ in range(n)] for _ in range(n)]
    queue = deque()
    queue.append((nowX, nowY))
    visited[nowX][nowY] = 0
    while queue:
        x, y = queue.popleft()
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]

            if 0 <= nx < n and 0 <= ny < n:
                if shark_size >= graph[nx][ny] and visited[nx][ny] == -1:
                    visited[nx][ny] = visited[x][y] + 1
                    queue.append((nx, ny))
    return visited

def solve(visited):
    # print(visited)
    x, y = 0, 0
    min_dis = INF
    for i in range(n):
        for j in range(n):
            if visited[i][j] != -1 and (1 <= graph[i][j] < shark_size):
                #잡아먹음
                if min_dis > visited[i][j]:
                    min_dis = visited[i][j]
                    x, y = i, j
    # print(min_dis)
    if min_dis == INF:
        #먹을게 없는 경우
        return False
    else:
        return x, y, min_dis

answer = 0
food = 0
while True:
    res = solve(bfs())
    # print(res)
    if not res: #False 라면 다 먹었다면
        print(answer)
        break
    else:
        nowX, nowY = res[0], res[1]
        answer += res[2]
        graph[nowX][nowY] = 0
        food += 1
    if food >= shark_size:
        shark_size += 1
        food = 0

#7568번 덩치
import sys
n = int(sys.stdin.readline())
nums = []
for _ in range(n):
    x, y = map(int, sys.stdin.readline().split())
    nums.append((x, y))

for i in range(n):
    res = 0
    nowX, nowY = nums[i]
    for j in range(n):
        a, b = nums[j]
        if i==j:
            continue
        if nowX < a and nowY < b:
            res += 1
    print(res + 1, end=' ')


#11866번 요세푸스 문제0
import sys
from collections import deque

n, k = map(int, sys.stdin.readline().split())
res = []
queue = deque()

for i in range(1, n+1):
    queue.append(i)

while queue:
    for _ in range(k-1):
        queue.append(queue.popleft())
    res.append(queue.popleft())

print("<", end="")
for i in range(len(res)):
    if i != len(res) - 1:
        print(res[i], end=", ")
    else:
        print(res[i], end="")
print(">")

#1158번 요세푸스 문제
import sys
from collections import deque

n, k = map(int, sys.stdin.readline().split())
res = []
queue = deque()

for i in range(1, n+1):
    queue.append(i)

while queue:
    for _ in range(k-1):
        queue.append(queue.popleft())
    res.append(queue.popleft())

print("<", end="")
for i in range(len(res)):
    if i != len(res) - 1:
        print(res[i], end=", ")
    else:
        print(res[i], end="")
print(">")

#1966번 프린터 큐
import sys
from collections import deque

testNum = int(sys.stdin.readline())
for _ in range(testNum):
    n, m = map(int, sys.stdin.readline().split())
    temp = list(map(int, sys.stdin.readline().split()))
    queue = deque()
    for i in range(len(temp)):
        queue.append((i, temp[i]))

    res = []
    # print(queue)
    while queue:
        #뒤에 큰게 있다면 맨뒤로 옮김
        flag = True
        for i in range(1, len(queue)):
            if queue[0][1] < queue[i][1]: # 뒤에가 더 크다면
                queue.append(queue.popleft())
                flag = False
                break
        if flag:
            res.append(queue.popleft())

        #없다면 res에 담아라
    # print(res)
    for i in range(len(res)):
        if m == res[i][0]:
            print(i+1)

#2167번 2차원 배열의 합
import sys
n, m = map(int, sys.stdin.readline().split())
graph = []
for _ in range(n):
    graph.append(list(map(int, sys.stdin.readline().split())))
k = int(sys.stdin.readline())
xy = []
for _ in range(k):
    xy.append(list(map(int, sys.stdin.readline().split())))

for i, j, x, y in xy:
    sum = 0
    for n in range(i-1, x):
        for m in range(j-1, y):
            sum += graph[n][m]
    print(sum)

#10866번 덱
import sys
from collections import deque
queue = deque()
n = int(sys.stdin.readline())
for _ in range(n):
    command = sys.stdin.readline().rstrip()
    # print(command)
    if "push_back" in command:
        queue.append(int(command[10:]))

    elif "push_front" in command:
        queue.insert(0, int(command[11:]))
    elif "pop_front" in command:
        if len(queue) == 0:
            print(-1)
        else:
            print(queue.popleft())
    elif "pop_back" in command:
        if len(queue) == 0:
            print(-1)
        else:
            print(queue.pop())
    elif "size" in command:
        print(len(queue))
    elif "empty" in command:
        if len(queue) == 0:
            print(1)
        else:
            print(0)
    elif "front" in command:
        if len(queue) == 0:
            print(-1)
        else:
            print(queue[0])
    elif "back" in command:
        if len(queue) == 0:
            print(-1)
        else:
            print(queue[len(queue) - 1])


#17413번 단어 뒤집기 2
import sys
S = sys.stdin.readline().strip() + ' '
stack = []
res = ""
cnt = 0
for i in S:
    if i == '<':
        cnt = 1
        for _ in range(len(stack)):
            res += stack.pop()
    stack.append(i)
    if i == '>':
        cnt = 0
        for _ in range(len(stack)):
            res += stack.pop(0)
    if i == ' ' and cnt == 0:
        stack.pop()
        for _ in range(len(stack)):
            res += stack.pop()
        res += ' '
print(res)

#2161번 카드1
import sys
from collections import deque
queue = deque()
n = int(sys.stdin.readline())
for i in range(1, n+1):
    queue.append(i)
res = []
while len(queue) > 1:
    res.append(queue.popleft())
    queue.append(queue.popleft())

for resData in res:
    print(resData, end=" ")
print(queue.popleft())

#2477번 참외밭
import sys
k = int(sys.stdin.readline())
arr = [list(map(int, sys.stdin.readline().split())) for _ in range(6)]
w, w_idx = 0, 0
h, h_idx = 0, 0
for i in range(6):
    if arr[i][0] == 1 or arr[i][0] == 2:
        if w < arr[i][1]:
            w = arr[i][1]
            w_idx = i
    elif arr[i][0] == 3 or arr[i][0] == 4:
        if h < arr[i][1]:
            h = arr[i][1]
            h_idx = i
smallRec = abs(arr[(w_idx-1)%6][1] - arr[(w_idx+1)%6][1]) * abs(arr[(h_idx-1)%6][1] - arr[(h_idx+1)%6][1])
bigRec = w*h
# print(w, h, w_idx, h_idx)
res = (bigRec - smallRec)*k
print(res)

#1051번 숫자 정사각형
import sys
n, m = map(int, sys.stdin.readline().split())
graph = []
for _ in range(n):
    graph.append(list(map(int, sys.stdin.readline().rstrip())))

minNum = min(n, m)
size = 1
res = 0
while size <= minNum:
    for row in range(n-size+1):
        for col in range(m-size+1):
            if graph[row][col] == graph[row+size-1][col+size-1] == graph[row][col+size-1] == graph[row+size-1][col]:
                res = max(res, size**2)
    size += 1

print(res)

#17608번 막대기
import sys
n = int(sys.stdin.readline())
graph = []
for _ in range(n):
    graph.append(int(sys.stdin.readline()))
# print(graph)
res = 1
maxNum = graph[n-1]
for i in range(n-2, -1, -1):
    # print(graph[i])
    if graph[i] > maxNum:
        maxNum = graph[i]
        res += 1
print(res)

#2669번 직사각형 네개의 합집합의 면적 구하기
import sys
graph = [[False for _ in range(100)] for _ in range(100)]
for _ in range(4):
    a, b, x, y = map(int, sys.stdin.readline().split())
    for i in range(a-1, x-1):
        for j in range(b-1, y-1):
            graph[i][j] = True

res = 0
for graphData in graph:
    res += graphData.count(True)
print(res)

#1138번 한 줄로 서기
import sys
n = int(sys.stdin.readline())
graph = list(map(int, sys.stdin.readline().split()))
res = [0 for _ in range(n)]
personNum = 1
for i in range(n):
    cnt = 0
    for j in range(n):
        if cnt == graph[i] and res[j] == 0:
            res[j] = i+1
            break
        elif res[j] == 0:
            cnt += 1
print(*res)

#14912번 숫자 빈도수
import sys
n, d = map(int, sys.stdin.readline().split())
dp = [0 for _ in range(10)]
#파이썬 숫자 자릿수
for i in range(1, n+1):
    while i != 0:
        dp[i % 10] += 1
        i = i // 10

print(dp[d])

#5635번 생일
import sys
n = int(sys.stdin.readline())
name = []
age = []
for _ in range(n):
    a, b, c, d = sys.stdin.readline().split()
    # b, c가 한자리이면 앞에 0 넣어줘라
    if len(b) == 1:
        b = '0' + b
    if len(c) == 1:
        c = '0' + c
    age.append((int(d+c+b), a))

age.sort(key=lambda x:x[0])
print(age[n-1][1])
print(age[0][1])

#11576번 Base Conversion
import sys
a, b = map(int, sys.stdin.readline().split())
m = int(sys.stdin.readline())
aNums = list(map(int, sys.stdin.readline().split()))

tenNum = 0
for i in range(m):
    tenNum += aNums[i]*(a**(m-1-i))
res = []
while tenNum > 0:
    res.append(tenNum % b)
    tenNum = tenNum // b

for j in range(len(res)-1, -1, -1):
    print(res[j], end = " ")

#10953번 A+B - 6
import sys
t = int(sys.stdin.readline())
for _ in range(t):
    a, b = map(int, sys.stdin.readline().split(','))
    print(a+b)

#10808번 알파벳 개수
import sys
s = sys.stdin.readline().rstrip()
res = [0] * 26
#아스키코드 변환 후 % 연산
for word in s:
    res[ord(word)%97] += 1
print(*res)