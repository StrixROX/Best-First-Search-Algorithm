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

    def __lt__(self, other):
        return False

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

# A* search algorithm (uniform cost)
def a_starViz(start, end):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {node.id: float('inf') for row in Nodes for node in row}
    g_score[start.id] = 0
    f_score = {node.id: float('inf') for row in Nodes for node in row}
    f_score[start.id] = h(start, end)

    open_set_hash = {start.id}

    while open_set.not_empty:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    quit()
        
        node = open_set.get()[2]
        open_set_hash.remove(node.id)

        if node == end:
            x = node
            while x != start:
                x.setState(6)
                x = came_from[x.id]
            
            end.setState(3)
            return True

        for i in node.neighbours:
            temp_g_score = g_score[node.id] + 1

            if temp_g_score < g_score[i.id]:
                came_from[i.id] = node
                g_score[i.id] = temp_g_score
                f_score[i.id] = temp_g_score + h(i, end)

                if i.id not in open_set_hash and not i.state == 1:
                    count += 1
                    open_set.put((f_score[i.id], count, i))
                    open_set_hash.add(i.id)
                    i.setState(4)
        
        draw()
    
    return False

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
        a_starViz(START_NODE, END_NODE)
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