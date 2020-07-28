# my utilities

from heapq import heapify, heappush, heappop

class PriorityQueue:
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heappop(self.heap)
        return item

    def is_empty(self):
        return len(self.heap) == 0

    def peek(self):
        return self.heap[0]

    def delete(self, item):
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                del self.heap[index]
                heapify(self.heap)
                return

    def update(self, item, priority):
        # priorityの更新
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapify(self.heap)
                return
        else:
            self.push(item, priority)
