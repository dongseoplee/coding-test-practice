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


#5430번 AC
import sys
from collections import deque

testNum = int(sys.stdin.readline())
for _ in range(testNum):
    errorYN = 'N'
    p = sys.stdin.readline()
    n = int(sys.stdin.readline())
    tempStr = sys.stdin.readline()
    if n != 0:
        str = deque(tempStr[1:len(tempStr) - 2].split(','))
    elif n == 0:
        str = deque()
    r = 0
    for i in range(len(p)):
        if p[i] == 'R':
            r += 1
        elif p[i] == 'D':
            if len(str) == 0:
                errorYN = 'Y'
            else:
                if r % 2 == 0:
                    str.popleft()
                else:
                    str.pop()

    if errorYN == 'N':
        if r % 2 == 0:
            print('[' + ",".join(str) + ']')
        else:
            str.reverse()
            print('[' + ",".join(str) + ']')

    else:
        print('error')

#4949번 균형잡힌 세상
import sys

while True:
    stack = []
    s = sys.stdin.readline().rstrip()
    flag = 0
    if s == '.':
        break
    for word in s:
        if word == '(':
            stack.append(word)
        elif word == '[':
            stack.append(word)
        elif word == ')':
            if len(stack) == 0 or stack[-1] == '[':
                print("no")
                flag = 1
                break
            else:
                stack.pop()
        elif word == ']':
            if len(stack) == 0 or stack[-1] == '(': # 비어 있거나 마지막에 다른 괄호라면
                print("no")
                flag = 1
                break
            else:
                stack.pop()
    if flag == 0:
        if len(stack) == 0:
            print("yes")
        else:
            print("no")