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

#14425번
import sys

n, m = map(int, sys.stdin.readline().split())
setList = []
inputList = []
for _ in range(n):
  setList.append(sys.stdin.readline().rstrip())
for _ in range(m):
  inputList.append(sys.stdin.readline().rstrip())

# print(inputList)
# print(setList)
#inputList 값 중 setList에 포함된 갯수
cnt = 0
for i in range(m):
  if inputList[i] in setList:
    cnt += 1

print(cnt)

#7785번 회사에 있는 사람
import sys
n = int(sys.stdin.readline())
res = dict()

for _ in range(n):
  name, status = sys.stdin.readline().split()

  if status == 'enter':
    res[name] = status
  else:
    del res[name]

#역순으로 출력
res = sorted(res.keys(), reverse=True)
for resData in res:
  print(resData)

#1620번
import sys

n, m = map(int, sys.stdin.readline().split())

dogam = []
question = []
pokeDict = dict()
pokeDict2 = dict()

for i in range(n):
  # dogam.append(sys.stdin.readline().rstrip())
  inputName = sys.stdin.readline().rstrip()
  pokeDict[i + 1] = inputName
  pokeDict2[inputName] = i + 1
for _ in range(m):
  question.append(sys.stdin.readline().rstrip())

# print(pokeDict)
# print(pokeDict2)

# 시간초과 딕셔너리 2개 만들어서 해결!!!
for i in range(m):
  try:
    question[i] = int(question[i])
  except:
    question[i]

for i in range(m):
  if type(question[i]) is int:
    print(pokeDict.get(question[i]))
  else:
    print(pokeDict2.get(question[i]))

#10816 숫자 카드 2
import sys
import copy

inputDict = dict()
resDict = dict()
n = int(sys.stdin.readline())
nList = list(map(int, sys.stdin.readline().split()))

for i in range(n):
  inputDict[nList[i]] = 0

m = int(sys.stdin.readline())
mList = (list(map(int, sys.stdin.readline().split())))
mListCopy = copy.deepcopy(mList)
mList = sorted(mList)

for j in range(m):
  resDict[mList[j]] = 0
# print(nList)
# print(mList)
# nList에서 검색후 resDict 키 값에 value + 1
for i in range(n):
  start = 0
  end = len(mList) - 1

  while start <= end:
    mid = (start + end) // 2
    if mList[mid] > nList[i]:
      end = mid - 1
    elif mList[mid] < nList[i]:
      start = mid + 1
    elif mList[mid] == nList[i]:  # 같으면 return이나 break로 while문 빠져나와야함!!
      # print(nList[i])
      resDict[nList[i]] += 1
      break

# print(resDict)
for i in range(m):
  print(resDict[mListCopy[i]], end=' ')

#1764번 듣보잡
import sys

n, m = map(int, sys.stdin.readline().split())  # n: 듣도 못한, m: 보도 못한
nList = []
mList = []
# print(n, m)
nDict = dict()
mDict = dict()
for i in range(n):
  nList.append(sys.stdin.readline().rstrip())
  # nDict[sys.stdin.readline().rstrip()] = 1
for j in range(m):
  # mList.append(sys.stdin.readline().rstrip())
  mDict[sys.stdin.readline().rstrip()] = 1

# print(nList, mList)
resList = []
# print(nList, mDict)
# 리스트와 딕셔너리 두개를 이용해서 중복되는 것 찾음!!
for nListData in nList:

  if nListData in mDict:
    # print(nListData)
    resList.append(nListData)

resList = sorted(resList)
print(len(resList))
for resListData in resList:
  print(resListData)

#1269번 대칭 차집합
import sys
aNum, bNum = map(int, sys.stdin.readline().split())

# print(aNum, bNum)
aSet = set(list(map(int, sys.stdin.readline().split())))

bSet = set(list(map(int, sys.stdin.readline().split())))

a = aSet - bSet
b = bSet - aSet
print(len(a) + len(b))

#11478번 서로 다른 부분 문자열의 개수
import sys
s = sys.stdin.readline()
resList = []
size = 0
for i in range(len(s)):
  size += 1
  for j in range(len(s) - size):
    resList.append(s[j:j+size])

# print(resList)
resSet = set(resList)
# print(resSet)
print(len(resSet))
