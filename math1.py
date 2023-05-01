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
