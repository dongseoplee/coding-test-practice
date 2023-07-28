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