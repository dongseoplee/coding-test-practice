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

#1436번 영화감독 슘
#다른 사람 풀이 문자열로 '666' 있으면 YN -> Y
import sys

n = int(sys.stdin.readline())
cnt = 0
firstSix = 666
while (1):

    firstSixList = list(str(firstSix))
    # print(firstSixList)
    # 연속된 6 3개 있다면 cnt + 1 하고 continue
    # 1의 자리 부터 3개씩 비교
    threeSixYN = 'N'
    for i in range(len(firstSixList) - 1, 1, -1):
        if firstSixList[i] == '6' and firstSixList[i - 1] == '6' and firstSixList[i - 2] == '6':
            threeSixYN = 'Y'

    # for문이 종료 후
    if threeSixYN == 'Y':
        cnt += 1

    # n번째 수와 카운트 된 횟수가 같다면 break
    if cnt == n:
        print(firstSix)
        break
    else:
        firstSix += 1

#2839번 설탕 배달
import sys

n = int(sys.stdin.readline())

xyList = []

for x in range(n // 5 + 1):
    for y in range(n // 3 + 1):
        if 5 * x + 3 * y == n:
            xyList.append(x + y)

if len(xyList) == 0:
    print(-1)
else:
    print(min(xyList))


