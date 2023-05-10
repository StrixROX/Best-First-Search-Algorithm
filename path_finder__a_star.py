# A* Search

import networkx as nx
import matplotlib.pyplot as plt
from queue import PriorityQueue
from typing import Union

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

# heuristic function: euclidean distance
def h(a:Node, b:Node) -> float:

    ax, ay = a.pos
    bx, by = b.pos

    return ((ax - bx)**2 + (ay-by)**2)**0.5

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
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 0, 0],
#     [0, 0, 0, 1, 0, 0],
#     [0, 0, 0, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0],
# ]

Map = [
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]

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

class GraphVisualization:
    def __init__(self, graph):
        # visual is a list which stores all 
        # the set of edges that constitutes a
        # graph
        self.visual = []
        self.graph = graph

    # addEdge function inputs the vertices of an
    # edge and appends it to the visual list
    def addEdge(self, a, b):
        temp = [a, b]
        self.visual.append(temp)

    # In visualize function G is an object of
    # class Graph given by networkx G.add_edges_from(visual)
    # creates a graph with a given list
    # nx.draw_networkx(G) - plots the graph
    # plt.show() - displays the graph
    def visualize(self):
        G = nx.Graph()
        G.add_edges_from(self.visual)

        cmap = []
        for node in G:
            if type(node) == tuple:
                j, i = node
            else:
                w = len(self.graph[0])

                j = node % w - 1
                i = (node - 1) // w

            if self.graph[i][j].isBlocked:
                cmap.append('red')
            else:
                cmap.append('blue')

        nx.draw_networkx(G,node_color=cmap)
        plt.show()

def showGraph(graph:list):
    w = len(graph[0])
    h = len(graph)

    G = GraphVisualization(graph)

    for i in range(h):
        for j in range(w):
            if j < w - 1:
                G.addEdge(graph[i][j].name, graph[i][j+1].name)
                # graph[i][j].connect(graph[i][j+1])
            if i < h - 1:
                G.addEdge(graph[i][j].name, graph[i+1][j].name)
                # graph[i][j].connect(graph[i+1][j])

    G.visualize()

Graph = generateGraph(Map)

def a_star(start:Node, end:Node) -> Union[list, None]:
    visited = []
    pq = PriorityQueue()
    pq.put((0, start)) # put the start node in pq to start search

    while pq.not_empty:
        cost, x = pq.get() # getting the node with least priority in pq and getting its cost(g(i)) and node itself
        visited.append(x) # add this node to the visited list

        print(x)

        # if this node the goal node then return the path taken
        if x == end:
            return visited

        # pick the node with lowest f(i) = g(i) + h(i) value
        # to traverse next and put it into pq
        y = list(x.neighbours)
        y.sort(key=lambda i: cost+h(i, end))
        pq.put((cost + h(y[0], end), y[0]))

    return None

print(a_star(start=Graph[0][0], end=Graph[-1][-1]))
showGraph(Graph)
