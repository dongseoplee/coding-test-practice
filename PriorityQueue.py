#11279번 최대 힙
import sys #y=-x 대칭을 이용해서 기본적으로 최소를 반환하는 것을 -1 곱해서 최대로 반환하는 것처럼 만든다.
import heapq
n = int(sys.stdin.readline())
nums = []
for _ in range(n):
    num = int(sys.stdin.readline())
    if num == 0:
        try:
            print(heapq.heappop(nums)*(-1))
        except:
            print(0)
    else:
        heapq.heappush(nums, num*(-1))

#1927번 최소 힙
import sys
import heapq
n = int(sys.stdin.readline())
nums = []
for _ in range(n):
    num = int(sys.stdin.readline())
    if num == 0:
        try:
            print(heapq.heappop(nums))
        except:
            print(0)
    else:
        heapq.heappush(nums, num)

#11286 절댓값 힙
import sys
import heapq
n = int(sys.stdin.readline())
nums = []
for _ in range(n):
    num = int(sys.stdin.readline())
    if num == 0:
        try:
            print(heapq.heappop(nums)[1])
        except:
            print(0)
    else:
        heapq.heappush(nums, (abs(num), num))

#1655번 가운데를 말해요
import sys
import heapq #힙 2개로 중간값 알아내기
n = int(sys.stdin.readline())
leftHeap = []
rightHeap = []
for _ in range(n):
    num = int(sys.stdin.readline())
    # 왼 오 길이 같으면 왼쪽에 넣는다
    # 왼쪽에서 최대값을 구해야 하므로 -1 곱해서 넣는다.
    if len(leftHeap) == len(rightHeap):
        heapq.heappush(leftHeap, num*(-1))
    else:
        heapq.heappush(rightHeap, num)
    #왼쪽 최소값*(-1)이 오른쪽 최소값보다 크다면 서로 바꾼다.
    if len(leftHeap) > 0 and len(rightHeap) > 0 and leftHeap[0]*(-1) > rightHeap[0]:
        leftPopNum = heapq.heappop(leftHeap)*(-1)
        rightPopNum = heapq.heappop(rightHeap)*(-1)
        heapq.heappush(leftHeap, rightPopNum)
        heapq.heappush(rightHeap, leftPopNum)
    print(leftHeap[0]*(-1)) #heap의 [0]은 최솟값이다.

