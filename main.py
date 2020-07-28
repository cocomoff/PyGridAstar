import numpy as np
import networkx as nx
import networkx.algorithms as nxa
import matplotlib.pyplot as plt
from itertools import product, count
from heapq import heappush, heappop

fn = "grid.txt"
M = list(map(lambda s: s.strip(), open(fn, "r").readlines()))
H = len(M)
W = len(M[0])

G = nx.Graph()
pos = {}
for (h, w) in product(range(H), range(W)):
    if M[h][w] != '#':
        G.add_node((h, w), loc=M[h][w])
        pos[(h, w)] = (w, h)

for (h, w) in product(range(H), range(W)):
    for (dh, dw) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nh, nw = h + dh, w + dw
        if nh < 0 or H <= dh or nw < 0 or W <= dw:
            continue
        if M[h][w] != '#' and M[nh][nw] != '#':
            G.add_edge((h, w), (nh, nw))


def astar_path(G, source, target, heuristic=None, weight='weight', W=1):
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
            return path, c, explored

        if curnode in explored:
            if explored[curnode] is None:
                continue
            qcost, h = enqueued[curnode]
            if qcost < dist:
                continue

        explored[curnode] = parent
        for neighbor, w in G[curnode].items():
            ncost = dist + w.get(weight, 1)
            if neighbor in enqueued:
                qcost, h = enqueued[neighbor]
                if qcost <= ncost:
                    continue
            else:
                h = heuristic(neighbor, target)

            # wAstar
            h = W * h
            enqueued[neighbor] = ncost, h
            push(queue, (ncost + h, next(c), neighbor, ncost, curnode))

    raise nx.NetworkXNoPath("Node %s not reachable from %s" % (target, source))

if __name__ == '__main__':
    # s = (3, 3)
    # g = (9, 12)
    nodes = list(G.nodes())
    ids = np.random.randint(0, len(nodes))
    idg = np.random.randint(0, len(nodes))
    s = nodes[ids]
    g = nodes[idg]

    def h(s1, s2):
        return abs(s1[0] - s2[0]) + abs(s1[1] - s2[1])

    for W in [0, 1, 2, 3]:
        apath, c, exp = astar_path(G, s, g, heuristic=h, W=W)
        exp_nodes = list(exp.keys())

        fig = plt.figure()
        ax = fig.gca()
        nx.draw(G, pos, ax=ax, node_color='k', node_size=20)
        for i in range(len(apath) - 1):
            pi = apath[i]
            pj = apath[i + 1]
            posI = pos[pi]
            posJ = pos[pj]
            ax.plot([posI[0], posJ[0]], [posI[1], posJ[1]], "b-")
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='r', alpha=0.8, node_size=30, nodelist=exp_nodes)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='b', alpha=0.8, node_size=30, nodelist=[s])
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='m', alpha=0.8, node_size=30, nodelist=[g])
        ax.set_title(c)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig("output/output_W{}.png".format(W), dpi=150)
        plt.close()