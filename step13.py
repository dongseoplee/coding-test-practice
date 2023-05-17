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

#1181번 단어 정렬
import sys
cnt = int(sys.stdin.readline())

resList = []
tempList = []
#글자수 계산해서 2중 리스트로 문자랑 같이 append
for _ in range(cnt):
  tempStr = sys.stdin.readline().rstrip()
  tempList.append(tempStr)

resList = list(set(tempList))
# print(resList)

finalList = []
for resData in resList:
  tempStoreList = []
  resDataLen = len(resData)
  tempStoreList.append(resDataLen)
  tempStoreList.append(resData)
  finalList.append(tempStoreList)

finalList.sort()
# print(finalList)

for i in range(len(finalList)):
  print(finalList[i][1])

#10814번 나이순 정렬
import sys
n = int(sys.stdin.readline())
#나이 가입순서 이름
resList = []
for i in range(n):
  tempList = []
  age, name = sys.stdin.readline().split()
  age = int(age)
  tempList.append(age)
  tempList.append(i)
  tempList.append(name)
  resList.append(tempList)

resList.sort()
#정렬후 출력은 나이와 이름만
for i in range(n):
  print(resList[i][0], end=' ')
  print(resList[i][2])

#18870번 좌표 압축
import sys
n = int(sys.stdin.readline())
inputList = list(map(int, sys.stdin.readline().split()))

setInputList = list(set(inputList))


setInputList.sort()
# print(setInputList)
# print(type(setInputList))

dict = {}

for i in range(len(setInputList)):
  dict[setInputList[i]] = i

# print(dict)
for inputListData in inputList:
  print(dict.get(inputListData), end=' ') #시간복잡도 O(1)
  # print(setInputList.index(inputListData), end=' ') #시간복잡도 O(n)


#11931번 수 정렬하기 4
import sys
n = int(sys.stdin.readline())
inputList = []
for _ in range(n):
  inputList.append(int(sys.stdin.readline()))

# print(inputList)
inputList.sort(reverse=True)
for inputListData in inputList:
  print(inputListData)


#15688번 수 정렬하기 5
import sys
n = int(sys.stdin.readline())
inputList = []
for _ in range(n):
  inputList.append(int(sys.stdin.readline()))

# print(inputList)
inputList.sort()
for inputListData in inputList:
  print(inputListData)