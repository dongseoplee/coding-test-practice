#24262번 알고리즘 수업 - 알고리즘의 수행 시간 1
print(1)
print(0)

#24263번 알고리즘 수업 - 알고리즘의 수행 시간 2
cnt = int(input())
print(cnt)
print(1)

#24264번 알고리즘 수업 - 알고리즘의 수행 시간 3
cnt = int(input())
print(cnt**2)
print(2)

#24265번 알고리즘 수업 - 알고리즘의 수행 시간 4
cnt = int(input())
print(int(cnt*(cnt-1)/2))
print(2)

#24266번 알고리즘 수업 - 알고리즘의 수행 시간 5
cnt = int(input())
print(cnt**3)
print(3)

#24267번 알고리즘 수업 - 알고리즘의 수행 시간 6
cnt = int(input())
sum = 0
for i in range(1, cnt - 1):
  sum = sum + i*(cnt-i-1)

print(sum)
print(3)

#24313번 알고리즘 수업 - 점근적 표기 1
a1, a0 = map(int, input().split())
c = int(input())
n0 = int(input())

n = n0
#c-a1이 0보다 크거나 같아야함
if (c-a1)*n >= a0 and c-a1 >= 0:
  print(1)
else:
  print(0)
