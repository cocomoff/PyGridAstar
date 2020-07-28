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


def bidirectional_mm_search(G, source, target, heuristic=None, weight='weight'):
    # Forward/Backwardの情報管理
    cur_state_f, path_f, g_f = source, [source], {source: 0}
    cur_state_b, path_b, g_b = target, [target], {target: 0}
    open_f = [(cur_state_f, path_f)]
    open_b = [(cur_state_b, path_b)]
    close_f, close_b = [], []
    U = float("inf") # これまでに見つかった最良解

    if heuristic is None:
        # def h(s, dir):
        #     return 0
        def h(s, dir):
            gg = target if dir == 0 else source
            return abs(s[0] - gg[0]) + abs(s[1] - gg[1])

    def search_dir(U, open1, open2, g1, g2, closed, dir):
        n, path = min_p_g(C, open1, g1, dir)
        open1.remove((n, path))
        closed.append((n, path))

        # implement using networkx instead of successor list
        for c, w in G[n].items():
            if found(open1, c) or found(closed, c):
                if g1[c] <= g1[n] + w.get(weight, 1):
                    continue
                open1 = delete(open1, c)

            g1[c] = g1[n] + w.get(weight, 1)
            open1.append((c, path + [c]))
            # 既に反対側から探索されて入っているかどうかをチェック
            if found(open2, c):
                U = min(U, g1[c] + g2[c])
        return U, open1, closed, g1

    def delete(open_list, n):
        for (c, path) in open_list:
            if c == n:
                open_list.remove((c, path))
        return open_list

    def found(open_list, n):
        for (c, path) in open_list:
            if c == n:
                return True
        return False

    def choose_min_n(open_list, g, dir):
        # dir (0=F/1=B) に関して，最良の状態を選択する
        prmin, prmin_f = float("inf"), float("inf")
        for (n, path) in open_list:
            f = g[n] + h(n, dir)
            pr = max(f, 2 * g[n])
            prmin = min(prmin, pr)
            prmin_f = min(prmin_f, f)
        return prmin, prmin_f, min(g.values())

    def min_p_g(prmin, open_list, g, dir):
        m = float("inf")
        node = target
        final_path = []
        for (n, path) in open_list:
            pr = max(g[n] + h(n, dir), 2 * g[n])
            if pr == prmin:
                if g[n] < m:
                    m = g[n]
                    node = n
                    final_path = path
        return node, final_path

    def get_path(open_f, open_b, verbose=False):
        if verbose:
            print("[OPEN_F]")
            for (n, pn) in open_f:
                print("  {} {}".format(n, pn))
            print("\n[OPEN_B]")
            for (n, pn) in open_b:
                print("  {} {}".format(n, pn))
            print()
        for (nf, path_f) in open_f:
            for (nb, path_b) in open_b:
                if nf == nb:
                    return path_f, path_b

    def opposite(path):
        pass

    # Bidirectional MM searchの本体 (両方のOpenが空じゃない限り探索)
    loop_counter = count()
    while open_f and open_b:
        prmin_f, fmin_f, gmin_f = choose_min_n(open_f, g_f, 0)
        prmin_b, fmin_b, gmin_b = choose_min_n(open_b, g_b, 1)
        C = min(prmin_f, prmin_b)

        if U <= max(C, fmin_f, fmin_b, gmin_f + gmin_b + 1):
            totalOpenNodes = len(open_f) + len(open_b) + 1
            totalClosedNodes = len(close_f) + len(close_b)
            print('\nTotal nodes expanded = {0}'.format(totalOpenNodes + totalClosedNodes))
            print(' (open nodes = {0} and closed nodes = {1})'.format(totalOpenNodes, totalClosedNodes))

            # print("\nPath length: {}".format(U))
            path_f, path_b = get_path(open_f, open_b, verbose=True)
            path = path_f + path_b[::-1][1:]
            return U, path

        # 次にどちらを開くか
        if C == prmin_f:
            U, open_f, close_f, g_f = search_dir(U, open_f, open_b, g_f, g_b, close_f, 0)
        else:
            U, open_b, close_b, g_b = search_dir(U, open_b, open_f, g_b, g_f, close_b, 1)

        if False:
            print(">{}".format(next(loop_counter)))
            print(prmin_f, fmin_f, gmin_f)
            print(prmin_b, fmin_b, gmin_b)
            print(C, U)
            print()