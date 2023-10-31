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