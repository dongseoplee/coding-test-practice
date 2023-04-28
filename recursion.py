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

#17478번 재귀함수가 뭔가요?
import sys

n = int(sys.stdin.readline())
print('어느 한 컴퓨터공학과 학생이 유명한 교수님을 찾아가 물었다.')
i = 0


def rec(i):
  print('____' * i + '\"재귀함수가 뭔가요?\"')
  if i != n:
    print('____' * i + '\"잘 들어보게. 옛날옛날 한 산 꼭대기에 이세상 모든 지식을 통달한 선인이 있었어.')
    print('____' * i + '마을 사람들은 모두 그 선인에게 수많은 질문을 했고, 모두 지혜롭게 대답해 주었지.')
    print('____' * i + '그의 답은 대부분 옳았다고 하네. 그런데 어느 날, 그 선인에게 한 선비가 찾아와서 물었어.\"')
  if i == n:
    print('____' * i + '\"재귀함수는 자기 자신을 호출하는 함수라네\"')

  if i == n:
    return 0
  else:
    i += 1
    rec(i)
    # return rec(i)가 아님 return 해버리면 다음 코드 실행하지 않음
    print('____' * i + '라고 답변하였지.')


rec(i)
print('라고 답변하였지.')

#27433번 팩토리얼2
import sys
n = int(sys.stdin.readline())
if n == 0:
  print(1)
  exit()
def fac(n):
  if n == 1: #재귀함수 탈출 조건이 있어야 함
    return 1
  else:
    return n * fac(n-1)

print(fac(n))