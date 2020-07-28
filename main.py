# -*- coding: utf-8 -*-

import time
import numpy as np
import networkx as nx
import networkx.algorithms as nxa
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from itertools import product, count
from heapq import heappush, heappop

from search import astar_path, astar_search

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


def sample_astar_path():
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
        apath, c, enq, exp = astar_path(G, s, g, heuristic=h, W=W)
        exp_nodes = list(exp.keys())
        enq_nodes = list(enq.keys())

        segs = []
        for i in range(len(apath) - 1):
            pi = apath[i]
            pj = apath[i + 1]
            posI = pos[pi]
            posJ = pos[pj]
            segs.append([(posI[0], posI[1]), (posJ[0], posJ[1])])
        lc = LineCollection(segs, color='k', lw=2.5, alpha=0.8)

        # plot
        fig = plt.figure()
        ax = fig.gca()
        nx.draw(G, pos, ax=ax, node_color='k', node_size=20)

        # OPEN
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='r', alpha=0.4, node_size=30, nodelist=enq_nodes)
        # CLOSE
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='r', alpha=0.8, node_size=30, nodelist=exp_nodes)
        # Start/Goal
        ax.plot([pos[s][0]], [pos[s][1]], color='b', marker='s')
        ax.plot([pos[g][0]], [pos[g][1]], color='m', marker='s')
        # Path
        ax.add_collection(lc)
        ax.set_title("{}/{}".format(len(exp_nodes), len(enq_nodes)))
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig("output/output_W{}.png".format(W), dpi=150)
        plt.close()


if __name__ == '__main__':
    nodes = list(G.nodes())
    ids = np.random.randint(0, len(nodes))
    idg = np.random.randint(0, len(nodes))
    s = nodes[ids]
    g = nodes[idg]
    print(s, g)

    def h(s1, s2):
        return abs(s1[0] - s2[0]) + abs(s1[1] - s2[1])
    
    for W in [0, 1, 2, 3]:
        t_start = time.time()
        cost, path = astar_search(G, s, g, heuristic=h, W=W)
        t_search = time.time() - t_start
        print(W, cost, t_search)
        print(">", path)