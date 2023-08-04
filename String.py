#8958번 OX퀴즈
import sys

testNum = int(sys.stdin.readline())
for _ in range(testNum):
    question = str(sys.stdin.readline().rstrip())
    # print(question)
    score = 0
    cnt = 0
    for i in range(len(question)):
        if question[i] == 'O':
            cnt += 1
            score += cnt
        if question[i] == 'X':
            cnt = 0

    print(score)

#11721번 열 개씩 끊어 출력하기
import sys
str1 = str(sys.stdin.readline().rstrip())

# print(str1)
cnt = len(str1) // 10  #몫
for i in range(cnt):
  print(str1[i*10:(i+1)*10])

print(str1[cnt*10:len(str1)])

#11719번 그대로 출력하기 2
import sys
for line in sys.stdin:
  print(line , end='')

#1259번 팰린드롬수
import sys

while (1):
    n = list(sys.stdin.readline().rstrip())
    if n[0] == '0':
        break

    flag = True
    for i in range(len(n) // 2):
        if n[i] != n[len(n) - i - 1]:
            flag = False
            break

    if flag:
        print('yes')
    else:
        print('no')

#다른 풀이
import sys

while (1):
    n = sys.stdin.readline().rstrip()
    if n == '0':
        break

    if n == n[::-1]:
        print('yes')
    else:
        print('no')