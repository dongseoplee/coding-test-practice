#11399번 ATM
import sys
n = int(sys.stdin.readline())
greedy = list(map(int, sys.stdin.readline().split()))
greedy = sorted(greedy)
sum = greedy[0]
for i in range(1, n):
  greedy[i] = greedy[i-1] + greedy[i]
  sum += greedy[i]


print(sum)

#11047번 동전 0
import sys
n, k = map(int, sys.stdin.readline().split())
price = [0] * n
cnt = [0] * n
for i in range(n):
  price[i] = int(sys.stdin.readline())

price.sort(reverse=True)

cnt = 0
#4500을 50000으로 나눈 몫은 0이다.
# 나눠지는 몫이 있으면 몫이 갯수고 k는 나머지 값으로 바뀐다.
for j in price:
  if k == 0:
    break
  cnt += k//j
  k %= j
print(cnt)

#1931번 회의실 배정
import sys #회의 끝나는 시간 기준으로 정렬, 회의가 빨리 끝나야 최대한 많은 회의 가능하다.
n = int(sys.stdin.readline())
conference = []
for i in range(n):
  conference.append(list(map(int, sys.stdin.readline().split())))
conference.sort(key=lambda x:(x[1], x[0]))
cnt = 1
endTime = conference[0][1]
for j in range(1, n):
  if conference[j][0] >= endTime:
    cnt+=1
    endTime = conference[j][1]

print(cnt)