class Node:
    def __init__(self, name:str):
        self.name = name
        self.next = []
    
    def connect(self, node, cost:int):
        self.next.append((node, cost))
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
    def show(self):
        print(f'Node: {self.name}, Next: {self.next}')


a = Node('A')
b = Node('B')
c = Node('C')
d = Node('D')
e = Node('E')
f = Node('F')
g = Node('G')

a.connect(b, 3)

b.connect(c, 1)
b.connect(d, 4)

d.connect(e, 1)

c.connect(e, 2)
c.connect(f, 1)

f.connect(g, 1)

g.connect(e, 2)

a.show()
b.show()
c.show()
d.show()
e.show()
f.show()
g.show()

