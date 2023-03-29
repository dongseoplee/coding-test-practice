#10845번 큐
import sys

num = int(sys.stdin.readline())
queue = []
front = 0
back = 0
res = []
for _ in range(num):
    command = list(input().split())
    if command[0] == 'push':
        queue.append(int(command[1]))
        back += 1
    if command[0] == 'pop':
        if front == back:
            # print(-1)
            res.append(-1)
        else:
            # print(queue[front])
            res.append(queue[front])

            front += 1
    if command[0] == 'size':
        # print(back - front)
        res.append(back - front)
    if command[0] == 'empty':
        if front == back:
            # print(1)
            res.append(1)
        else:
            # print(0)
            res.append(0)
    if command[0] == 'front':
        if front == back:
            # print(-1)
            res.append(-1)
        else:
            # print(queue[front])
            res.append(queue[front])
    if command[0] == 'back':
        if front == back:
            # print(-1)
            res.append(-1)
        else:
            # print(queue[back-1])
            res.append(queue[back - 1])

for i in range(len(res)):
    print(res[i])
