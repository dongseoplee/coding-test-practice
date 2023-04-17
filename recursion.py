#1629번 곱셈
#분할 정복의 원리 Divide and Conquer(DAC)
import sys
a, b, c = map(int, sys.stdin.readline().split())
# a를 c로 나눈 나머지 의 b제곱
# a승 계산 -> 2a, 2a + 1승 계산
def mod(a, b, c):
  if b == 1:
    return a % c
  elif b % 2 == 0:
    return (mod(a, b//2, c)**2) % c
  else:
    return ((mod(a, b//2, c)**2)*a) % c

print(mod(a, b, c))


