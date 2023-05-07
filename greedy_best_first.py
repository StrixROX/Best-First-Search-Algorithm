# Greedy Best-First Search

from queue import PriorityQueue
from typing import Union

# graph node definition
class Node:
    def __init__(self, name:str, pos:tuple) -> None:
        self.name = name
        self.pos = pos
        self.neighbours = []

    def connect(self, node) -> None:
        if node not in self.neighbours:
            self.neighbours.append(node)

        if self not in node.neighbours:
            node.neighbours.append(self)

    def __str__(self) -> str:
        return self.name + repr(self.pos)

    def __repr__(self) -> str:
        return self.name

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

# defining graph nodes
a = Node('A', (0, 0))
b = Node('B', (5, 0))
c = Node('C', (8, 0))
d = Node('D', (5, 1))
e = Node('E', (8, 1))
f = Node('F', (4, 1))
g = Node('G', (4, 3))

# defining graph edges
a.connect(b)

b.connect(c)
b.connect(d)

d.connect(e)
d.connect(f)

c.connect(e)

f.connect(g)

g.connect(e)

# displaying graph node details
a.show()
b.show()
c.show()
d.show()
e.show()
f.show()
g.show()

def bestFirstSearch(start:Node, end:Node) -> Union[list, None]:
    visited = []
    pq = PriorityQueue()
    pq.put((0, start)) # put the start node in pq to start search

    while pq.not_empty:
        x = pq.get()[1] # getting the node with least priority in pq
        visited.append(x) # add this node to the visited list

        # if this node the goal node then return the path taken
        if x == end:
            return visited

        # iterate through all the unvisited nodes connected to this node
        # and put it into pq using f(i) = h(i)
        for i in x.neighbours:
            if i not in visited:
                pq.put((h(i, end), i))

    # if the goal node is not connected to the start node, return None
    return None

print(bestFirstSearch(start=a, end=e))