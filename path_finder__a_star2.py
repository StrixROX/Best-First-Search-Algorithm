import pygame
from typing import Union
from queue import PriorityQueue

pygame.init()

# Window dimensions
WIDTH, HEIGHT = (800, 800)

# Grid dimensions
ROWS, COLS = (50,50)

# Create the game window
window = pygame.display.set_mode((WIDTH, HEIGHT))

# Set the title of the window
pygame.display.set_caption("A* Search Based Path Planning")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
DARKGREY = (150, 150, 150)
GREEN = (20, 217, 72)
RED = (255, 0, 0)
TURQUOISE = (64, 224, 208)
PURPLE = (160, 32, 240)

START_SEARCH = 0
CLICK_MODE = 0
# 0: creating / clearing obstacles
# 1: creating / clearing start and goal nodes

START_NODE = None
END_NODE = None

class Node:
    def __init__(self, _id, pos):
        self.id = _id
        self.pos = pos
        self.neighbours = []
        self.state = 0
        self.g = float('inf')
        self.h = float('inf')
        self.f = float('inf')
        # 0: empty, traversable node
        # 1: blocked, non-traversable node
        # 2: start node
        # 3: goal node
        # 4: searching node for path
        # 5: searching node neighbour for path
        # 6: path found
        
        self.color_map = {
            0: WHITE,
            1: BLACK,
            2: GREEN,
            3: RED,
            4: DARKGREY,
            5: TURQUOISE,
            6: PURPLE,
        }
        
    def connect(self, node):
        if node not in self.neighbours:
            self.neighbours.append(node)

        if self not in node.neighbours:
            node.neighbours.append(self)
    
    def setState(self, to):
        self.state = to if to in [0, 1, 2, 3, 4, 5, 6] else self.state

    def getColor(self):
        return self.color_map[self.state]

    def __str__(self):
        # return self.id + repr(self.pos)
        return str(self.id)# + repr(tuple(self.neighbours))

    def __repr__(self):
        return str(self.id)
        # return self.id + repr(tuple(self.neighbours))

    def __eq__(self, __value):
        if not isinstance(__value, Node):
            return False
        if self.pos != __value.pos:
            return False

        return True

    def __hash__(self):
        return hash((self.id,) + self.pos)

    def show(self):
        # print(f'Node: {self.id}, Pos: {self.pos}, Neighbours: {self.neighbours}')
        x, y = self.pos

        pygame.draw.rect(window, self.color_map[self.state], (x, y, WIDTH / COLS, HEIGHT / ROWS), 1)

Nodes = [[Node(i*COLS + j+1, (j, i)) for j in range(COLS)] for i in range(ROWS)]

def drawGrid():
    for r in range(ROWS - 1):
        pygame.draw.line(window, GREY, (0, (r+1)*HEIGHT/ROWS), (WIDTH, (r+1)*HEIGHT/ROWS))

    for c in range(COLS - 1):
        pygame.draw.line(window, GREY, ((c+1)*WIDTH/COLS, 0), ((c+1)*WIDTH/COLS, HEIGHT))

def draw():
    cWidth = WIDTH / COLS
    cHeight = HEIGHT / ROWS

    for i in range(ROWS):
        for j in range(COLS):
            pygame.draw.rect(window, Nodes[i][j].getColor(), (i*cWidth, j*cHeight, (i + 1)*cWidth, (j + 1)*cHeight), 0)

    drawGrid()

    # Update display
    pygame.display.update()

def handleClicks():
    global START_NODE, END_NODE
    if pygame.mouse.get_pressed()[0]: # LEFT
        y, x = pygame.mouse.get_pos()
        cWidth = WIDTH / COLS
        cHeight = HEIGHT / ROWS

        i = min(max(int(y // cHeight), 0), ROWS - 1)
        j = min(max(int(x // cWidth), 0), COLS - 1)

        node = Nodes[i][j]

        if CLICK_MODE == 0:
            if node.state != 1:
                node.setState(1)
        elif CLICK_MODE == 1:
            if node.state != 2:
                if START_NODE is not None:
                    START_NODE.setState(0)
                START_NODE = node
                START_NODE.setState(2)
    elif pygame.mouse.get_pressed()[2]: # RIGHT
        y, x = pygame.mouse.get_pos()
        cWidth = WIDTH / COLS
        cHeight = HEIGHT / ROWS

        i = min(max(int(y // cHeight), 0), ROWS - 1)
        j = min(max(int(x // cWidth), 0), COLS - 1)

        node = Nodes[i][j]

        if CLICK_MODE == 0:
            if node.state != 0:
                node.setState(0)
        elif CLICK_MODE == 1:
            if node.state != 3:
                if END_NODE is not None:
                    END_NODE.setState(0)
                END_NODE = node
                END_NODE.setState(3)

# heuristic function: manhattan distance with obstacle detection
def h(a:Node, b:Node) -> float:
    # a to b

    ax, ay = a.pos
    bx, by = b.pos

    # dist = ((ax - bx)**2 + (ay-by)**2)**0.5
    dist = abs(ax - bx) + abs(ay - by)

    if a.state == 1:
        return float('inf')

    return dist

# cost function: manhattan distance
def g(a:Node, b:Node) -> float:
    ax, ay = a.pos
    bx, by = b.pos

    return abs(ax - bx) + abs(ay - by)

# recursive A* search algorithm (uniform cost)
# def a_starViz(start, end, visited = []) -> Union[list, None]:
#     start.setState(4)
#     draw()
#     if start == end:
#         # reached goal
#         visited.append(end)

#         for i in visited[::-1]:
#             i.setState(6)
#             draw()

#         return visited
#     if h(start,end) == float('inf'):
#         # reached obstacle
#         return None

#     # only consider the neighbours that are not blocked
#     temp = list(filter(lambda x: h(x, end) != float('inf'), start.neighbours))
#     # sort them by heuristic function only since it is uniform cost
#     # otherwise there would be a separate cost function
#     # added to the heuristic function value
#     temp.sort(key=lambda x: h(x, end))

#     for i in temp:
#         i.setState(5)
#         draw()
#         if i not in visited:
#             res = a_starViz(i, end, visited + [start])
#             if res is not None:
#                 # if we found the goal float it up the recursion chain
#                 # and return the path to main program
#                 return res
#         i.setState(4)
#         draw()

#     # if no path could be found to goal
#     return None

# def a_starViz(start, end):
#     count = 0
#     visited = []
#     pq = PriorityQueue()
#     pq.put((0, count, start))

#     while pq.not_empty:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 quit()

#             if event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_q:
#                     quit()

#         cost, _, node = pq.get()
#         if node in visited:
#             continue
#         visited.append(node)
#         node.setState(4)
#         # print(f"Added {node} to visited. {visited}")
#         # print('4 |>', node)

#         if node == end:
#             for i in visited:
#                 i.setState(6)
#                 # print('6 |>', i)
#             return visited
        
#         for i in node.neighbours:
#             if i not in visited:
#                 # print('\t', i, pq.queue)
#                 count += 1
#                 f_score = cost+g(node, i) + h(i, end)
#                 if f_score == float('inf'):
#                     continue
#                 pq.put((f_score, count, i))
#                 i.setState(5)

#         draw()

def a_starViz(start, end):
    count = 0
    open_set = PriorityQueue()
    open_set_hash = {}
    came_from = {}
    closed_set = []

    start.g = 0
    start.h = h(start, end)
    start.f = start.g + start.h

    open_set.put((0, count, start))
    open_set_hash[start] = (0, count, start)
    came_from[start] = None

    while len(open_set.queue) != 0:
        print(open_set.queue, len(open_set.queue))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    quit()
        
        node = open_set.get()[2]

        if node == end:
            while came_from[node] is not None:
                node = came_from[node]
                node.setState(6)
            end.setState(3)
            return True
        
        closed_set.append(node)

        for i in node.neighbours:
            if i.state == 1:
                continue
            
            _g = node.g + g(node, i)
            _h = h(i, end)
            _f = _g + _h
            
            print("checking", i, _f)

            if _f < i.f:
                came_from[i] = node

                if not (i in open_set.queue or i in closed_set):
                    count += 1
                    i.g, i.h, i.f = _g, _h, _f
                    open_set.put((_f, count, i))
                    open_set_hash[i] = (_f, count, i)
                    i.setState(4)

                if i in open_set.queue:
                    hash = open_set_hash[i]
                    i.g, i.h, i.f = _g, _h, _f
                    open_set.put((_f, hash[1], i))
                    open_set_hash[i] = (_f, hash[1], i)

                if i in closed_set:
                    ind = closed_set.index(i)
                    closed_set = closed_set[:ind] + closed_set[ind+1:]

                    hash = open_set_hash[i]
                    i.g, i.h, i.f = _g, _h, _f
                    open_set.put((_f, hash[1], i))
                    open_set_hash[i] = (_f, hash[1], i)
        
        draw()

    return False

# def a_starViz(start, end):
#     count = 0
#     open_set = PriorityQueue()
#     open_set.put((0, count, start))
#     came_from = {}
#     g_score = {node: float('inf') for row in Nodes for node in row}
#     g_score[start] = 0
#     f_score = {node: float('inf') for row in Nodes for node in row}
#     f_score[start] = h(start, end)

#     open_set_hash = {start}

#     while open_set.not_empty:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 quit()

#             if event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_q:
#                     quit()
        
#         node = open_set.get()[2]
#         open_set_hash.remove(node)

#         if node == end:
#             x = node
#             while x != start:
#                 x.setState(6)
#                 x = came_from[x]
            
#             end.setState(3)
#             return True

#         for i in node.neighbours:
#             temp_g_score = g_score[node] + 1

#             if temp_g_score < g_score[i]:
#                 came_from[i] = node
#                 g_score[i] = temp_g_score
#                 f_score[i] = temp_g_score + h(i, end)

#                 if i not in open_set_hash and not i.state == 1:
#                     count += 1
#                     open_set.put((f_score[i], count, i))
#                     open_set_hash.add(i)
#                     i.setState(4)
        
#         draw()
    
#     return False

def setup():
    # temp
    global START_NODE, END_NODE
    START_NODE = Nodes[0][0]
    END_NODE = Nodes[-1][-1]
    
    START_NODE.setState(2)
    END_NODE.setState(3)

    # Make node connections
    for i in range(ROWS):
        for j in range(COLS):
            if j < COLS - 1:
                Nodes[i][j].connect(Nodes[i][j + 1])
            if i < ROWS - 1:
                Nodes[i][j].connect(Nodes[i+1][j])

def loop():
    global START_SEARCH

    # Clear the window
    window.fill(WHITE)

    handleClicks()

    if START_SEARCH:
        print(a_starViz(START_NODE, END_NODE))
        START_SEARCH = 0

    draw()

def GameLoop():
    global CLICK_MODE, START_SEARCH

    setup()
    # Game loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LSHIFT:
                    CLICK_MODE = 1
                elif event.key == pygame.K_SPACE:
                    START_SEARCH = 1
            
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LSHIFT:
                    CLICK_MODE = 0
                elif event.key == pygame.K_SPACE:
                    START_SEARCH = 0

        loop()

    pygame.quit()
GameLoop()