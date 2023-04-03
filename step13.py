#2750 수 정렬하기
import sys
numList = []
num = int(sys.stdin.readline())

for _ in range(num):
  inputNum = int(sys.stdin.readline())
  numList.append(inputNum)


setNumList = list(set(numList))
setNumList.sort()

for setNumListData in setNumList:
  print(setNumListData)

#2587번 대표값2
import sys
numList = []

for _ in range(5):
  numList.append(int(sys.stdin.readline()))

print(int(sum(numList)/len(numList)))
numList.sort()
print(numList[2])

#25305 커트라인
import sys
n, k = map(int, sys.stdin.readline().split())

numList = list(map(int, sys.stdin.readline().split()))

numList.sort(reverse=True)
print(numList[k-1])

#2751번 수 정렬하기 2
import sys
n = int(sys.stdin.readline())
numList = []
for _ in range(n):
  numList.append(int(sys.stdin.readline()))

numList.sort()

for numListData in numList:
  print(numListData)

#10989번 수 정렬하기 3
import sys
n = int(sys.stdin.readline())
numList = [0] * 10001

for _ in range(n):
  inputNum = int(sys.stdin.readline())
  numList[inputNum] += 1

for i in range(10001):
  if numList[i] != 0:
    for _ in range(numList[i]):
      print(i)

#1427번 소트인사이드
import sys
n = sys.stdin.readline().rstrip()
nList = list(n)
nListInt = []
for i in range(len(nList)):
  nListInt.append(int(nList[i]))

nListInt.sort(reverse=True)

for i in range(len(nListInt)):
  print(nListInt[i], end='')

#11650번 좌표 정렬하기
import sys
cnt = int(sys.stdin.readline())

numList = []
# print(cnt)
for _ in range(cnt):
  inputList = list(map(int, sys.stdin.readline().split()))
  # print(inputList)
  numList.append(inputList)

#자동으로 1차원 순으로 동일 1차원에서는 2차원 순으로 정렬됨
numList.sort()

for i in range(cnt):
  print(numList[i][0], numList[i][1])


#11651번 좌표 정렬하기 2
import sys
cnt = int(sys.stdin.readline())
numList = []

for _ in range(cnt):
  n1, n2 = map(int, sys.stdin.readline().split())
  tempList = []
  tempList.append(n2)
  tempList.append(n1)
  numList.append(tempList)

numList.sort()

for i in range(cnt):
  print(numList[i][1], numList[i][0])

