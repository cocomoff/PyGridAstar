import numpy as np
import networkx as nx
import networkx.algorithms as nxa
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from itertools import product, count
from util import PriorityQueue
from heapq import heappush, heappop

def astar_path(G, source, target, heuristic=None, weight='weight', W=1):
    # taken from networkx library
    push, pop = heappush, heappop
    if heuristic is None:
        def heuristic(u, v):
            return 0

    # Counter
    c = count()
    queue = [(0, next(c), source, 0, None)]

    # Open/Close
    enqueued = {}
    explored = {}
    while queue:
        _, _, curnode, dist, parent = pop(queue)

        if curnode == target:
            path = [curnode]
            node = parent
            while node is not None:
                path.append(node)
                node = explored[node]
            path.reverse()
            return path, c, enqueued, explored

        if curnode in explored:
            # 親がない場合
            if explored[curnode] is None:
                continue
            # 親がある場合のコストを取ってきて比較
            # より小さいコストで過去に訪問してたらいらない
            qcost, _ = enqueued[curnode]
            if qcost < dist:
                continue

        # 展開済みとして親を記録する
        explored[curnode] = parent
        for neighbor, w in G[curnode].items():
            new_cost = dist + w.get(weight, 1)
            if neighbor in enqueued:
                queue_cost, _ = enqueued[neighbor]
                if queue_cost <= new_cost:
                    continue
            else:
                h = heuristic(neighbor, target)

            # wAstar
            h = W * h
            enqueued[neighbor] = new_cost, h
            priority = new_cost + h
            push(queue, (priority, next(c), neighbor, new_cost, curnode))


def astar_search(G, source, target, heuristic=None, weight='weight', W=1.0):
    path = []
    visited = set({})
    cur_cost = 0
    cur_state = source

    if heuristic is None:
        def heuristic(u, v):
            return 0

    # Priority queueの初期化
    pQueue = PriorityQueue()
    init_priority = heuristic(source, target)
    init_entry = (cur_state, [], cur_cost)
    pQueue.push(init_entry, init_priority)

    while not pQueue.is_empty():
        if cur_state == target:
            path.append(target)
            break
        if cur_state not in visited:
            visited.add(cur_state)
            for neighbor, w in G[cur_state].items():
                new_cost = cur_cost + w.get(weight, 1)
                new_priority = new_cost + W * heuristic(neighbor, target)
                entry = (neighbor, path + [cur_state], new_cost)
                pQueue.update(entry, new_priority)

        (cur_state, path, cur_cost) = pQueue.pop()
    return cur_cost, path