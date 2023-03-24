#27323번 직사각형
A = int(input())
B = int(input())

print(A*B)

#1085번 직사각형에서 탈출
x, y, w, h = map(int, input().split())
res = []
res.append(x)
res.append(y)
res.append(h-y)
res.append(w-x)

print(min(res))

#3009번
xList = []
yList = []
answer = []
for i in range(3):
    x1, y1 = map(int, input().split())
    xList.append(x1)
    yList.append(y1)

setx = list(set(xList))
sety = list(set(yList))
for i in range(2):
    if xList.count(setx[i]) == 1:
        answer.append(setx[i])

for i in range(2):
    if yList.count(sety[i]) == 1:
        answer.append(sety[i])

print(*answer)

#15894번 수학은 체육과목 입니다
cnt = int(input())
print(4*cnt)

#9063번 대지
x = []
y = []

cnt = int(input())

for i in range(cnt):
  x1, y1 = map(int, input().split())
  x.append(x1)
  y.append(y1)

width = abs(max(x) - min(x))
height = abs(max(y) - min(y))

print(width*height)

#10101번 삼각형 외우기
angle = []
for i in range(3):
  angle.append(int(input()))

angleSet = set(angle)
#set을 이용하자 동일 값-> 리스트를 set으로 변경
if sum(angle) == 180:
  if angle.count(60) == 3:
    print('Equilateral')
  elif len(angleSet) == 2:
    print('Isosceles')
  elif len(angleSet) == 3:
    print('Scalene')
else:
  print('Error')

#5073번 삼각형과 세 변
while(1):
  lengthList = list(map(int, input().split()))
  if sum(lengthList) == 0:
    break
  #삼각형 조건 충족
  else:
    if sum(lengthList) > 2*max(lengthList):
      setLength = set(lengthList)
      if len(setLength) == 1:
        print('Equilateral')
      elif len(setLength) == 2:
        print('Isosceles')
      elif len(setLength) == 3:
        print('Scalene')
    else:
      print('Invalid')

#14215번 세 막대
lenList = list(map(int, input().split()))
lenList.sort()

if 2 * max(lenList) >= sum(lenList):
  res = (lenList[0] + lenList[1]) * 2 - 1
  print(res)

else:
  print(sum(lenList))
