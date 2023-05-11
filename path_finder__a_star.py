# A* Search

import networkx as nx
import matplotlib.pyplot as plt
from queue import PriorityQueue
from typing import Union

# n = 6
# Map = [[0 for i in range(n)] for j in range(n)]

# Map = [
#     [0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,1,1,1,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0]
# ]

# Map = [
#     [0, 0, 0, 1, 0, 0],
#     [0, 0, 0, 1, 0, 0],
#     [0, 0, 0, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 0, 0],
#     [0, 0, 0, 1, 0, 0],
# ]

Map = [
    [0, 0, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 0, 0],
    [0, 1, 1, 0]
]

# graph node definition
class Node:
    def __init__(self, name:int, pos:tuple, isBlocked:bool = False) -> None:
        self.name = name
        self.pos = pos
        self.isBlocked = isBlocked
        self.neighbours = []

    def connect(self, node) -> None:
        if node not in self.neighbours:
            self.neighbours.append(node)

        if self not in node.neighbours:
            node.neighbours.append(self)

    def __str__(self) -> str:
        # return self.name + repr(self.pos)
        return str(self.name) + repr(tuple(self.neighbours))

    def __repr__(self) -> str:
        return str(self.name)
        # return self.name + repr(tuple(self.neighbours))

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Node):
            return False
        if self.pos != __value.pos:
            return False

        return True

    def show(self) -> None:
        print(f'Node: {self.name}, Pos: {self.pos}, Neighbours: {self.neighbours}')

def generateGraph(Map:list) -> list:
    w = len(Map[0])
    h = len(Map)

    graphNodes = Map

    for i in range(h):
        for j in range(w):
            temp = Node(i*w + j+1, (j, i), Map[i][j] == 1)
            graphNodes[i][j] = temp
    
    for i in range(h):
        for j in range(w):
            if j < w - 1:
                graphNodes[i][j].connect(graphNodes[i][j+1])
            if i < h - 1:
                graphNodes[i][j].connect(graphNodes[i+1][j])

    return graphNodes

def showGraph(graph:list):
    w = len(graph[0])
    h = len(graph)

    visibleNodes = []

    for i in range(h):
        for j in range(w):
            if j < w - 1:
                visibleNodes.append([graph[i][j].name, graph[i][j+1].name])
                # G.addEdge(graph[i][j].name, graph[i][j+1].name)
                # graph[i][j].connect(graph[i][j+1])
            if i < h - 1:
                visibleNodes.append([graph[i][j].name, graph[i+1][j].name])
                # G.addEdge(graph[i][j].name, graph[i+1][j].name)
                # graph[i][j].connect(graph[i+1][j])

    G = nx.Graph()
    G.add_edges_from(visibleNodes)

    cmap = []
    for node in G:
        if type(node) == tuple:
            j, i = node
        else:
            w = len(graph[0])

            j = node % w - 1
            i = (node - 1) // w

        if graph[i][j].isBlocked:
            cmap.append('red')
        else:
            cmap.append('blue')

    plt.figure('Graph Visualisation')
    nx.draw_networkx(G,node_color=cmap)
    # plt.show()

# heuristic function: euclidean distance with obstacle detection
def h(a:Node, b:Node) -> float:
    # a to b

    ax, ay = a.pos
    bx, by = b.pos

    dist = ((ax - bx)**2 + (ay-by)**2)**0.5

    if a.isBlocked:
        return float('inf')

    return dist

def a_star(start:Node, end:Node, visited:list = [], cost:float = 0) -> Union[list, None]:
    if start == end:
        return visited + [end]

    temp = list(start.neighbours)
    temp = list(filter(lambda x: h(x, end) != float('inf'), start.neighbours))
    temp.sort(key=lambda x: h(x, end))

    for i in temp:
        if i not in visited:
            res = a_star(i, end, visited + [start], cost + h(i, end))
            if res is not None:
                return res

    return None

def showHeuristicMap(graph:list):
    hViz = []
    goal = graph[-1][-1]

    for i in graph:
        temp = []
        for j in i:
            temp.append(h(j, goal))
        hViz.append(temp)
        # print(temp)

    plt.figure('Heuristic Function Heatmap')
    plt.title('Heuristic Function, h(i)')
    plt.xlabel('X-Axis')
    plt.ylabel('Y-Axis')
    plt.imshow(hViz, cmap='Wistia', interpolation='nearest')
    # plt.show()

Graph = generateGraph(Map)
start = Graph[0][0]
end = Graph[-1][-1]

path = a_star(start, end)
print(path, 'Cost:', len(path or []))
showGraph(Graph)
showHeuristicMap(Graph)

plt.show()