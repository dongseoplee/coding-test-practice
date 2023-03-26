#2798번 블랙잭
n, m = map(int, input().split())
card = list(map(int, input().split()))

# print(n, m, card)
cardlength = len(card)
max = 0
for i in range(cardlength):
  for j in range(i + 1, cardlength):
    for k in range(j + 1, cardlength):
      if max <= card[i] + card[j] + card[k] and card[i] + card[j] + card[k] <= m:
        max = card[i] + card[j] + card[k]
        # print(card[i], card[j], card[k])
      else:
        max = max

print(max)

#2231번 분해합
num = int(input())
#자릿수를 더하기 위해 정수형을 문자열로 변환
existYN = 'N'
for i in range(1, num + 1):
  numList = list(map(int, str(i))) #정수 i를 문자열로 변환 -> map함수를 통해 int형으로 변환 최종적으로 list형으로 변환
  if num == sum(numList) + i:
    print(i)
    existYN = 'Y'
    break

if existYN == 'N':
  print(0)