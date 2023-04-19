#1934번 최소공배수
import sys

testNum = int(sys.stdin.readline())
for _ in range(testNum):
    a, b = map(int, sys.stdin.readline().split())
    multi = a * b

    # 유클리드 호제법
    # 최소공배수 = 두 자연수의 곱 / 최대공약수
    # 최대공약수는 유클리드 호제법

    inputList = []
    inputList.append(a)
    inputList.append(b)
    inputList.append(a % b)
    # print(inputList)
    i = 0
    while (1):
        if inputList[i + 2] == 0:
            GCD = inputList[i + 1]
            break
        else:
            inputList.append(inputList[i + 1])
            inputList.append(inputList[i + 2])
            mod = inputList[i + 1] % inputList[i + 2]
            inputList.append(mod)
            i += 3

    print(int(a * b / GCD))

#13241번 최소공배수
import sys

a, b = map(int, sys.stdin.readline().split())
multi = a * b

# 유클리드 호제법
# 최소공배수 = 두 자연수의 곱 / 최대공약수
# 최대공약수는 유클리드 호제법


inputList = []
inputList.append(a)
inputList.append(b)
inputList.append(a % b)
# print(inputList)
i = 0
while (1):
    if inputList[i + 2] == 0:
        GCD = inputList[i + 1]
        break
    else:
        inputList.append(inputList[i + 1])
        inputList.append(inputList[i + 2])
        mod = inputList[i + 1] % inputList[i + 2]
        inputList.append(mod)
        i += 3

print(int(a * b / GCD))

#1735번 분수 합
import sys
a1, b1 = map(int, sys.stdin.readline().split())
a2, b2 = map(int, sys.stdin.readline().split())

bm = b1 * b2
bj = a1*b2 + a2*b1

# print(bj, bm)

def GCD(a, b):
  if a%b == 0:
    return b
  else:
    return GCD(b, a%b) #함수 내에서 본인 함수 호출할때 return 써야한다.

resGCD = GCD(bm, bj) # 21 35
print(int(bj/resGCD), end = ' ')
print(int(bm/resGCD), end = ' ')
