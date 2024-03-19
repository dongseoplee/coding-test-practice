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

#2744번 대소문자 바꾸기
import sys
# sys.stdin = open("input.txt", "r")
str = sys.stdin.readline().rstrip()
# print(ord(str[0]))
ans = ''
# print(ord('Z'), ord('z'))
for char in str:
    if 65 <= ord(char) <= 90: #대문자
        ans += chr(ord(char) + 32)
    elif 97 <= ord(char) <= 122: #소문자
        ans += chr(ord(char) - 32)
print(ans)

#2754번 학점계산
import sys
set = {'A+': 4.3, 'A0': 4.0, 'A-': 3.7, 'B+': 3.3, 'B0': 3.0, 'B-': 2.7,
       'C+': 2.3, 'C0': 2.0,'C-': 1.7,
'D+': 1.3,'D0': 1.0,'D-': 0.7,
'F': 0.0}

# sys.stdin = open("input.txt", "r")
score = sys.stdin.readline().rstrip()
print(set[score])

#9012번 괄호
import sys
from collections import deque
# sys.stdin = open("input.txt", "r")

n = int(sys.stdin.readline())
for _ in range(n):
    q = deque()
    flag = True
    temp = sys.stdin.readline().rstrip()
    for a in temp:
        if a == '(':
            q.append('(')
        elif a == ')':
            if len(q) == 0:
                q.append(')')
                break
            else:
                q.pop()
    if len(q) != 0:
        print("NO")
    else:
        print("YES")


#1100번 하얀 칸
import sys
from collections import deque
# sys.stdin = open("input.txt", "r")

graph = [list(sys.stdin.readline().rstrip()) for _ in range(8)]
ans = 0
for i in range(8):
    if i % 2 == 0:
        s = 0
    else:
        s = 1
    for j in range(s, 8, 2):
            if graph[i][j] == 'F':
                ans += 1
print(ans)

#9003번 단어 뒤집기
import sys
# sys.stdin = open("input.txt", "r")
T = int(sys.stdin.readline())
for _ in range(T):
    lst = list(sys.stdin.readline().split())
    ans = ""
    for lst_str in lst:
        ans += lst_str[::-1] + " "
    print(ans)

#15829번 Hashing
import sys
# sys.stdin = open("input.txt", "r")
L = int(sys.stdin.readline())
temp = sys.stdin.readline().rstrip()
ans = 0
sum = 0
for i in range(len(temp)):
    sum += (ord(temp[i]) - 96)*(31**i)
ans = sum % 1234567891
print(ans)

#11656번 접미사 배열
import sys
# sys.stdin = open("input.txt", "r")

temp = sys.stdin.readline().rstrip()
ans = []
for i in range(len(temp)):
    ans.append(temp[i:])
ans.sort()
for ansData in ans:
    print(ansData)

