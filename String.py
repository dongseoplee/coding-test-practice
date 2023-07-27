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