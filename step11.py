#2798번 블랙잭
n, m = map(int, input().split())
card = list(map(int, input().split()))

# print(n, m, card)
cardlength = len(card)
max = 0
for i in range(cardlength):
  for j in range(i + 1, cardlength):
    for k in range(j + 1, cardlength):
      if max <= card[i] + card[j] + card[k] and card[i] + card[j] + card[k] <= m:
        max = card[i] + card[j] + card[k]
        # print(card[i], card[j], card[k])
      else:
        max = max

print(max)

#2231번 분해합
num = int(input())
#자릿수를 더하기 위해 정수형을 문자열로 변환
existYN = 'N'
for i in range(1, num + 1):
  numList = list(map(int, str(i))) #정수 i를 문자열로 변환 -> map함수를 통해 int형으로 변환 최종적으로 list형으로 변환
  if num == sum(numList) + i:
    print(i)
    existYN = 'Y'
    break

if existYN == 'N':
  print(0)

#19532번 수학은 비대면강의입니다.
a, b, c, d, e, f = map(int, input().split())
#완전탐색, 브루트포스 범위값 모두 대입
for x in range(-999, 1000):
  for y in range(-999, 1000):
    if a*x + b*y == c and d*x + e*y == f:
      print(x, y)


#1018번 체스판 다시 칠하기 (미해결)
n, m = map(int, input().split())
chessBoard = []
cntList = []
for i in range(n):
  inputStr = input()
  chessBoard.append(list(inputStr))

for row in range(n - 8 + 1):  # 012
  for col in range(m - 8 + 1):  # 012345
    cnt = 0
    if chessBoard[row][col] == 'W':
      for i in range(8):
        if i % 2 == 0:
          for j in range(0, 8, 2):
            if chessBoard[row + i][col + j] != 'W':
              cnt += 1

          for k in range(1, 8, 2):
            if chessBoard[row + i][col + k] != 'B':
              cnt += 1
        if i % 2 != 0:
          for j in range(0, 8, 2):
            if chessBoard[row + i][col + j] != 'B':
              cnt += 1

          for k in range(1, 8, 2):
            if chessBoard[row + i][col + k] != 'W':
              cnt += 1

    if chessBoard[row][col] == 'B':
      for i in range(8):
        if i % 2 == 0:
          for j in range(0, 8, 2):
            if chessBoard[row + i][col + j] != 'B':
              cnt += 1

          for k in range(1, 8, 2):
            if chessBoard[row + i][col + k] != 'W':
              cnt += 1
        if i % 2 != 0:
          for j in range(0, 8, 2):
            if chessBoard[row + i][col + j] != 'W':
              cnt += 1

          for k in range(1, 8, 2):
            if chessBoard[row + i][col + k] != 'B':
              cnt += 1
    cntList.append(cnt)

print(cntList)
print(min(cntList))


#1018번 체스판 다시 칠하기 (해결)
import sys

n, m = map(int, sys.stdin.readline().split())
board = [sys.stdin.readline().strip() for _ in range(n)]
count = []

for a in range(n-7):
    for b in range(m-7):
        index1 = 0  # W로 시작할 경우 : 바꿔야 할 체스 판 개수
        index2 = 0  # B로 시작할 경우 : 바꿔야 할 체스 판 개수
      #w로 시작할때와 b로 시작할때 비교해서 작은 값 count 리스트에 저장
        for i in range(a, a+8):
            for j in range(b, b+8):
                if (i+j) % 2 == 0:
                    if board[i][j] != 'W':
                        index1 += 1
                    if board[i][j] != 'B':
                        index2 += 1
                else:
                    if board[i][j] != 'B':
                        index1 += 1
                    if board[i][j] != 'W':
                        index2 += 1
        count.append(min(index1, index2))
print(min(count))
