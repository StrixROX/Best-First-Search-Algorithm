# Uniform Cost(1) A* Search

import networkx as nx
import matplotlib.pyplot as plt
from typing import Union

### Change 'Map' to any rectangular coordinate map
### Change 'start' and 'end' to any nodes in the constructed 'Graph'
### Run to see Uniform Cost(1) A* Search in action

# n = 6
# Map = [[0 for i in range(n)] for j in range(n)]

# Map = [
#     [0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,1,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,1,1,1,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0],
#     [0,0,1,0,0,1,0,0,0,0],
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

# duplicate for graphing purposes
# ignore
MAP = []
for i in Map:
    temp = []
    for j in i:
        temp.append(j)
    MAP.append(temp)

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

# heuristic function: euclidean distance with obstacle detection
def h(a:Node, b:Node) -> float:
    # a to b

    ax, ay = a.pos
    bx, by = b.pos

    dist = ((ax - bx)**2 + (ay-by)**2)**0.5

    if a.isBlocked:
        return float('inf')

    return dist

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

def showGraph(graph:list) -> None:
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

def showGraphPath(graph:list, path:list) -> None:
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
        elif graph[i][j] in path:
            cmap.append('green')
        else:
            cmap.append('blue')

    plt.figure('Path Visualisation')
    nx.draw_networkx(G,node_color=cmap)
    # plt.show()

# recursive A* search algorithm (uniform cost)
def a_star(start:Node, end:Node, visited:list = []) -> Union[list, None]:
    if start == end:
        # reached goal
        return visited + [end]
    if h(start,end) == float('inf'):
        # reached obstacle
        return None

    # only consider the neighbours that are not blocked
    temp = list(filter(lambda x: h(x, end) != float('inf'), start.neighbours))
    # sort them by heuristic function only since it is uniform cost
    # otherwise there would be a separate cost function
    # added to the heuristic function value
    temp.sort(key=lambda x: h(x, end))

    for i in temp:
        if i not in visited:
            res = a_star(i, end, visited + [start])
            if res is not None:
                # if we found the goal float it up the recursion chain
                # and return the path to main program
                return res

    # if no path could be found to goal
    return None

def showHeuristicMap(graph:list) -> None:
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

    _w = len(graph[0])
    _h = len(graph)

    for i in range(_w):
        for j in range(_h):
            plt.text(j, i, graph[i][j].name,
                ha="center", va="center", color="black")

def showMap(m:list) -> None:
    plt.figure("Rectangular Coordinate Map Visualisation")
    plt.imshow(m, cmap="binary")

    _w = len(m[0])
    _h = len(m)

    for i in range(_w):
        for j in range(_h):
            plt.text(j, i, i*_w + j+1,
                ha="center", va="center", color="red")

def showMapPath(m:list, path:list) -> None:
    plt.figure("Rectangular Coordinate Path Visualisation")

    _w = len(m[0])
    _h = len(m)

    for i in range(_w):
        for j in range(_h):
            plt.text(j, i, i*_w + j+1,
                ha="center", va="center", color="green")
    
    for node in path:
        j, i = node.pos
        m[i][j] = 0.5

    plt.imshow(m, cmap="binary")

Graph = generateGraph(Map)
start = Graph[0][0]
end = Graph[-1][-1]

Path = a_star(start, end)
print(Path, 'Cost:', len(Path or []))

showGraph(Graph)
showHeuristicMap(Graph)
if Path is not None:
    showGraphPath(Graph, Path)

showMap(MAP)
showMapPath(MAP, Path)

plt.show()