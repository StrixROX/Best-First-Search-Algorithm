import pygame
from queue import PriorityQueue
import pickle

pygame.init()

BOUNDING_BOX_SCALING_FACTOR = 100
BOUNDING_BOX_DATA_LOC = 'out'
with open(BOUNDING_BOX_DATA_LOC, 'rb') as f:
    bbox_all = pickle.load(f)

# Environment dimensions - 10m x 10m
DIMENSION_X, DIMENSION_Y = (10, 10) # meters
WIDTH, HEIGHT = (DIMENSION_X * BOUNDING_BOX_SCALING_FACTOR, DIMENSION_Y * BOUNDING_BOX_SCALING_FACTOR) # 1px = 1cm

# Grid dimensions
ROWS, COLS = (100,100)

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

    def __hash__(self):
        return hash((self.id,) + self.pos)

    def show(self):
        # print(f'Node: {self.id}, Pos: {self.pos}, Neighbours: {self.neighbours}')
        x, y = self.pos

        pygame.draw.rect(window, self.color_map[self.state], (x, y, WIDTH / COLS, HEIGHT / ROWS), 1)

Nodes = [[Node(i*COLS + j+1, (j, i)) for j in range(COLS)] for i in range(ROWS)]

def scalarMultiply(vector, scalar):
    temp = map(lambda x: x * scalar, vector)

    return type(vector)(temp)

def scalarAdd(vector, scalar):
    temp = map(lambda x: x + scalar, vector)

    return type(vector)(temp)

def vectorAdd(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Attempting to add vectors of different shapes")

    out = []
    for i in range(len(vector1)):
        out.append(vector1[i] + vector2[i])

    return tuple(out)

def dotProduct(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Attempting to multiply vectors of different shapes")

    out = []
    for i in range(len(vector1)):
        out.append(vector1[i] * vector2[i])

    return tuple(out)

def distance(p1, p2):
	x1, y1 = p1
	x2, y2 = p2

	return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def drawNodes():
    cWidth = WIDTH / COLS
    cHeight = HEIGHT / ROWS

    for i in range(ROWS):
        for j in range(COLS):
            pygame.draw.rect(window, Nodes[i][j].getColor(), (i*cWidth, j*cHeight, (i + 1)*cWidth, (j + 1)*cHeight), 0)

def drawGrid():
    for r in range(ROWS - 1):
        pygame.draw.line(window, GREY, (0, (r+1)*HEIGHT/ROWS), (WIDTH, (r+1)*HEIGHT/ROWS))

    for c in range(COLS - 1):
        pygame.draw.line(window, GREY, ((c+1)*WIDTH/COLS, 0), ((c+1)*WIDTH/COLS, HEIGHT))

def drawBoundingBoxes():
    global BOUNDING_BOX_SCALING_FACTOR
    for bbox in bbox_all:
        # for v in bbox:
        #     x = v[0] * scalingFactor
        #     y = v[1] * scalingFactor
        #     pygame.draw.circle(window, TURQUOISE, ((x + 0.5) * WIDTH / COLS + WIDTH/2, (y + 0.5) * HEIGHT / ROWS + HEIGHT/2), 5, 0)

        for i in range(4): # because here bounding box contains 4 vertices
            x1, y1 = vectorAdd(scalarMultiply(bbox[i], BOUNDING_BOX_SCALING_FACTOR), (WIDTH/2, HEIGHT/2))
            for j in range(i+1, 4): # with this we basically get all combinations of the 4 vertices, ie, 6 combinations
                x2, y2 = vectorAdd(scalarMultiply(bbox[j], BOUNDING_BOX_SCALING_FACTOR), (WIDTH/2, HEIGHT/2))

                pygame.draw.line(window, TURQUOISE, (x1, y1), (x2, y2), 1)

def getEdgeLineFunction(v1, v2):
    # edge line function: f(x,y) = y - mx - c
    x1, y1 = v1
    x2, y2 = v2

    def f(x,y):
        if x1 == x2:
            return -(x - x1)
        return (y - y1) - ((y2 - y1) / (x2 - x1))*(x - x1)
    
    return f

def pnpoly(nvert, vertx, verty, testx, testy):
    c = 0
    for i in range(nvert):
        j = (nvert - 1) if i == 0 else (i - 1)
        if ((verty[i] > testy) != (verty[j] > testy)) and (testx < ((vertx[j] - vertx[i]) * (testy - verty[i]) / (verty[j] - verty[i]) + vertx[i])):
            c = not c 
    return c

def traingleArea(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    return abs(0.5 * (x1*(y2 - y3) + x2*(y3-y1) + x3*(y1-y2)))

def mutateNodesUsingBoundingBox(bbox):
    # 1. close the nodes in which the vertices lie
    cWidth = WIDTH / COLS
    cHeight = HEIGHT / ROWS
    for v in bbox:
        x, y = vectorAdd(scalarMultiply(v, BOUNDING_BOX_SCALING_FACTOR), (WIDTH/2, HEIGHT/2))

        j = int(y // cHeight)
        i = int(x // cWidth)
        # coordinates are flipped fsr... idk

        if i < len(Nodes) and j < len(Nodes[0]):
            Nodes[i][j].setState(1)

    # 2. close the nodes that are inside or intersecting with the bounding box
    A = vectorAdd(scalarMultiply(bbox[0], BOUNDING_BOX_SCALING_FACTOR), (WIDTH/2, HEIGHT/2))

    others = bbox[1:]
    others.sort(key=lambda x: distance(x, A))

    B = vectorAdd(scalarMultiply(others[0], BOUNDING_BOX_SCALING_FACTOR), (WIDTH/2, HEIGHT/2))
    C = vectorAdd(scalarMultiply(others[2], BOUNDING_BOX_SCALING_FACTOR), (WIDTH/2, HEIGHT/2))
    D = vectorAdd(scalarMultiply(others[1], BOUNDING_BOX_SCALING_FACTOR), (WIDTH/2, HEIGHT/2))

    Xs = []
    Ys = []
    for i in [A, B, C, D]:
        Xs.append(i[0])
        Ys.append(i[1])

    Imin = int(min(Ys) // cHeight)
    Imax = int(max(Ys) // cHeight)
    Jmin = int(min(Xs) // cWidth)
    Jmax = int(max(Xs) // cWidth)

    A_ABCD = traingleArea(A, B, C) + traingleArea(A, D, C)

    def isPointInsideBoundingBox(x, y):
        # Area check
        P = (x, y)
        A_AB = traingleArea(P, A, B)
        A_BC = traingleArea(P, B, C)
        A_CD = traingleArea(P, C, D)
        A_AD = traingleArea(P, A, D)

        A_sum = A_AB + A_BC + A_CD + A_AD

        if A_sum <= A_ABCD:
            return True

        # pnpoly check
        if pnpoly(4, Xs, Ys, x, y):
            return True

        return False

    for I in range(Imin, Imax + 1):
        for J in range(Jmin, Jmax + 1):
            if I >= len(Nodes) or J >= len(Nodes[0]):
                continue

            if Nodes[J][I].state == 1:
                continue
            # Nodes[J][I].setState(2)
            x, y = dotProduct((J + 0.5, I + 0.5), (cWidth, cHeight))

            # reduces checks for points that are "more" inside
            # ie, greater overlap
            if isPointInsideBoundingBox(x, y):
                Nodes[J][I].setState(1)
                continue

            # if that comes out as false,
            # see if any of the corners of the cell
            # is inside the bounding box
            atLeastOneInside = False
            for i in [0, 1]:
                if atLeastOneInside:
                    break
                for j in [0, 1]:
                    x, y = dotProduct((J + j, I + i), (cWidth, cHeight))
                    atLeastOneInside = isPointInsideBoundingBox(x, y)
                    if atLeastOneInside:
                        break
            if atLeastOneInside:
                Nodes[J][I].setState(1)

def draw():
    for bbox in bbox_all:
        mutateNodesUsingBoundingBox(bbox)

    # draw cells/nodes
    drawNodes()

    # draw bounding boxes from environment
    drawBoundingBoxes()

    # draw the grid
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

# heuristic function: euclidean distance
def h(a:Node, b:Node) -> float:
    # a to b
    ax, ay = a.pos
    bx, by = b.pos

    dist = ((ax - bx)**2 + (ay-by)**2)**0.5
    # dist = abs(ax - bx) + abs(ay - by)

    if a.state == 1:
        return float('inf')

    return dist

# cost function: euclidean distance
def g(a:Node, b:Node) -> float:
    ax, ay = a.pos
    bx, by = b.pos

    dist = ((ax - bx)**2 + (ay-by)**2)**0.5
    # dist = abs(ax - bx) + abs(ay - by)

    return dist

def a_star(start, end):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {node: float('inf') for row in Nodes for node in row}
    g_score[start] = 0
    f_score = {node: float('inf') for row in Nodes for node in row}
    f_score[start] = h(start, end)

    open_set_hash = {start}

    while open_set.not_empty:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    quit()
        
        node = open_set.get()[2]
        open_set_hash.remove(node)

        if node == end:
            x = node
            while x != start:
                x.setState(6)
                x = came_from[x]
            
            end.setState(3)
            return True

        for i in node.neighbours:
            temp_g_score = g_score[node] + 1

            if temp_g_score < g_score[i]:
                came_from[i] = node
                g_score[i] = temp_g_score
                f_score[i] = temp_g_score + h(i, end)

                if i not in open_set_hash and not i.state == 1:
                    count += 1
                    open_set.put((f_score[i], count, i))
                    open_set_hash.add(i)
                    # uncomment to mark searched nodes
                    # i.setState(4)

        # uncomment to render each step of the algorithm
        # draw()
    
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

def printPath():
    a = []
    for i in Nodes:
        for j in i:
            if j.state == 6:
                a.append(j.pos)

    print(a)

def loop():
    global START_SEARCH

    # Clear the window
    window.fill(WHITE)

    handleClicks()

    if START_SEARCH:
        # with open('./ded', 'rb') as f:
        #     path = pickle.load(f)
        # for i in range(len(path)):
        #     for j in range(len(path[i])):
        #         Nodes[i][j].setState(path[i][j][1])
        # draw()
        pathFound = a_star(START_NODE, END_NODE)
        if pathFound:
            printPath()
            draw()
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