#10815번 숫자카드
import sys
n = int(sys.stdin.readline())
cardList = list(map(int, sys.stdin.readline().split()))
m = int(sys.stdin.readline())
inputList = list(map(int, sys.stdin.readline().split()))

#딕셔너리 이용
cardDict = {}
for i in range(len(cardList)):
  cardDict[cardList[i]] = 0


for i in range(m):
  if inputList[i] in cardDict:
    print(1, end=' ')
  else:
    print(0, end=' ')

#이진 탐색 이용
import sys
n = int(sys.stdin.readline())
cardList = sorted(list(map(int, sys.stdin.readline().split())))
m = int(sys.stdin.readline())
inputList = list(map(int, sys.stdin.readline().split()))


for i in range(len(inputList)):
  exist_YN = 'N'
  start = 0
  end = n - 1
  # print(i)
  # 이진탐색
  while start <= end:
    mid = (start + end) // 2
    if cardList[mid] == inputList[i]:
      exist_YN = 'Y'
      break ###### 찾으면 break 써야함!!
    elif cardList[mid] < inputList[i]:
      start = mid + 1
    elif cardList[mid] > inputList[i]:
      end = mid - 1

  if exist_YN == 'N':
    print(0, end=' ')
  else:
    print(1, end=' ')