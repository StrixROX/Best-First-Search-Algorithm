import pygame
from queue import PriorityQueue
import pickle
import random

pygame.init()

BOUNDING_BOX_SCALING_FACTOR = 65
BOUNDING_BOX_DATA_LOC_ROBOT = '/mnt/d/out_robot'
BOUNDING_BOX_DATA_LOC = '/mnt/d/out_boundingBoxes'
OBSTACLE_DATA_LOC_ALL = '/mnt/d/out_obstaclesAll'
OBSTACLE_DATA_LOC_MOVING = '/mnt/d/out_obstaclesMoving'
OBSTACLE_DATA_LOC_STATIC = '/mnt/d/out_obstaclesStatic'
OBSTACLE_DATA_LOC_TREES = '/mnt/d/out_obstaclesTrees'

POS_DATA_LOC_ROBOT = '/mnt/d/out_robotPos'
POS_DATA_LOC_TARGET = '/mnt/d/out_targetPos'

ROBOT_BOUNDING_BOX_SIZE = 0

with open(BOUNDING_BOX_DATA_LOC_ROBOT, 'rb') as f:
    bbox_robot = pickle.load(f)
with open(BOUNDING_BOX_DATA_LOC, 'rb') as f:
    bbox_all = pickle.load(f)
with open(OBSTACLE_DATA_LOC_ALL, 'rb') as f:
    obstacles_all = pickle.load(f)
with open(OBSTACLE_DATA_LOC_MOVING, 'rb') as f:
    obstacles_moving = pickle.load(f)
with open(OBSTACLE_DATA_LOC_STATIC, 'rb') as f:
    obstacles_static = pickle.load(f)
with open(OBSTACLE_DATA_LOC_TREES, 'rb') as f:
    obstacles_trees = pickle.load(f)

with open(POS_DATA_LOC_ROBOT, 'rb') as f:
    robot_pos = pickle.load(f)
with open(POS_DATA_LOC_TARGET, 'rb') as f:
    target_pos = pickle.load(f)

# with open(BOUNDING_BOX_DATA_LOC, 'rb') as f:
#     bbox_all = pickle.load(f)
# with open(BOUNDING_BOX_DATA_LOC_ROBOT, 'rb') as f:
#     bbox_robot = pickle.load(f)
# with open(POS_DATA_LOC_ROBOT, 'rb') as f:
#     robotPos = pickle.load(f)
# with open(POS_DATA_LOC_TARGET, 'rb') as f:
#     targetPos = pickle.load(f)

# Environment dimensions - 15m x 15m
DIMENSION_X, DIMENSION_Y = (15, 15) # meters
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
    return
    for r in range(ROWS - 1):
        pygame.draw.line(window, GREY, (0, (r+1)*HEIGHT/ROWS), (WIDTH, (r+1)*HEIGHT/ROWS))

    for c in range(COLS - 1):
        pygame.draw.line(window, GREY, ((c+1)*WIDTH/COLS, 0), ((c+1)*WIDTH/COLS, HEIGHT))
 
obstacles_visible = [i for i in obstacles_static if i in obstacles_trees.keys()]
# obstacles_visible = [i for i in obstacles_static if i in list(obstacles_trees.keys())]

def getCorrectBBoxLimits(handle):
    bbox_combined = []
    for j in obstacles_trees[handle]:
        bbox_combined += bbox_all[j]

    Xs = [p[0] for p in bbox_combined]
    Ys = [p[1] for p in bbox_combined]

    return min(Xs), max(Xs), min(Ys), max(Ys)

def getCorrectBBox(handle):
    xmin, xmax, ymin, ymax = getCorrectBBoxLimits(handle)

    bbox_new = [(i, j) for i in [xmin, xmax] for j in [ymin, ymax]]

    return bbox_new

def resetScene():
    global bbox_all, bbox_robot
    global obstacles_all, obstacles_moving, obstacles_static, obstacles_trees
    global robot_pos, target_pos

    global obstacles_visible

    for i in obstacles_visible:
        bbox_all[i] = getCorrectBBox(i)

    # rearrange visible obstacles
    lowX = (-1) * DIMENSION_X / 2
    highX = DIMENSION_X / 2
    lowY = (-1) * DIMENSION_Y / 2
    highY = DIMENSION_Y / 2

    done = {}

    def isOverlapping(limits1, limits2):
        x1, X1, y1, Y1 = limits1
        x2, X2, y2, Y2 = limits2

        if (x1 <= x2 <= X1) or (x1 <= X2 <= X1):
            if (y1 <= y2 <= Y1) or (y1 <= Y2 <= Y1):
                return True
        
        return False

    def displaceObstacle(obs, delX, delY):
        for i in range(len(bbox_all[obs])):
            bbox_all[obs][i] = vectorAdd(bbox_all[obs][i], (delX, delY))

    for obs in obstacles_trees.keys():
        if obs not in obstacles_static:
            continue

        if obs in done.keys():
            continue

        bbox = bbox_all

    for obs in obstacles_trees.keys():
        if obs not in obstacles_static:
            continue

        if obs in done.keys():
            continue

        bbox = bbox_all[obs]
        Xs = [p[0] for p in bbox]
        Ys = [p[1] for p in bbox]

        xmin, xmax, ymin, ymax = min(Xs), max(Xs), min(Ys), max(Ys)
        xmid = (xmax + xmin)/2
        ymid = (ymax + ymin)/2

        trying = True
        while trying:
            delX = random.uniform(lowX + abs(xmid - xmin), highX - abs(xmax - xmid)) - xmid
            delY = random.uniform(lowY + abs(ymid - ymin), highY - abs(ymax - ymid)) - ymid

            newLimits = [xmin+delX, xmax+delX, ymin+delY, ymax+delY]
            
            trying = False
            for i in done:
                if isOverlapping(done[i], newLimits) or isOverlapping(newLimits, done[i]):
                    trying = True
                    break
            
            if not trying:
                done[obs] = newLimits
                displaceObstacle(obs, delX, delY)

# # temp check
# for i in range(len(obstacles_visible)):
#     bbox = bbox_all[obstacles_visible[i]]
#     Xs = [p[0] for p in bbox]
#     Ys = [p[1] for p in bbox]

#     xmin, xmax, ymin, ymax = min(Xs), max(Xs), min(Ys), max(Ys)

#     limits1 = [xmin, xmax, ymin, ymax]
#     for j in range(i, len(obstacles_visible)):
#         _bbox = bbox_all[obstacles_visible[j]]
#         _Xs = [p[0] for p in _bbox]
#         _Ys = [p[1] for p in _bbox]

#         _xmin, _xmax, _ymin, _ymax = min(_Xs), max(_Xs), min(_Ys), max(_Ys)

#         limits2 = [_xmin, _xmax, _ymin, _ymax]

#         check1 = isOverlapping(limits1, limits2)
#         check2 = isOverlapping(limits2, limits1)
#         if (check1 or check2) and i != j:
#             print(check1, check2)
#             print(obstacles_visible[i], obstacles_visible[j])

def drawBoundingBoxes():
    global BOUNDING_BOX_SCALING_FACTOR
    global obstacles_visible

    # making bounding boxes of all obstacles + the robot
    # for obs in obstacles_visible:
    for obs in obstacles_static:
        # for v in bbox:
        #     x = v[0] * scalingFactor
        #     y = v[1] * scalingFactor
        #     pygame.draw.circle(window, TURQUOISE, ((x + 0.5) * WIDTH / COLS + WIDTH/2, (y + 0.5) * HEIGHT / ROWS + HEIGHT/2), 5, 0)

        bbox = bbox_all[obs]

        # because here bounding box contains 4 vertices
        for i in range(4):
            x1, y1 = vectorAdd(scalarMultiply(bbox[i], BOUNDING_BOX_SCALING_FACTOR), (WIDTH/2, HEIGHT/2))
            
            # with this we basically get all combinations of the 4 vertices, ie, 6 combinations
            for j in range(i+1, 4):
                x2, y2 = vectorAdd(scalarMultiply(bbox[j], BOUNDING_BOX_SCALING_FACTOR), (WIDTH/2, HEIGHT/2))

                pygame.draw.line(window, TURQUOISE, (x1, y1), (x2, y2), 1)

    # show robot bbox
    bbox = bbox_robot

    # because here bounding box contains 4 vertices
    for i in range(4):
        x1, y1 = vectorAdd(scalarMultiply(bbox[i], BOUNDING_BOX_SCALING_FACTOR), (WIDTH/2, HEIGHT/2))
        
        # with this we basically get all combinations of the 4 vertices, ie, 6 combinations
        for j in range(i+1, 4):
            x2, y2 = vectorAdd(scalarMultiply(bbox[j], BOUNDING_BOX_SCALING_FACTOR), (WIDTH/2, HEIGHT/2))

            pygame.draw.line(window, GREEN, (x1, y1), (x2, y2), 1)

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

        # if i < len(Nodes) and j < len(Nodes[i]):
        #     Nodes[i][j].setState(1)
        if 0 <= i < len(Nodes):
            if 0 <= j < len(Nodes[i]):
                Nodes[i][j].setState(1)

    # 2. close the nodes that are inside or intersecting with the bounding box
    A = vectorAdd(scalarMultiply(bbox[0], BOUNDING_BOX_SCALING_FACTOR), (WIDTH/2, HEIGHT/2))

    others = bbox[1:]
    others.sort(key=lambda x: distance(x, bbox[0]))

    B = vectorAdd(scalarMultiply(others[0], BOUNDING_BOX_SCALING_FACTOR), (WIDTH/2, HEIGHT/2))
    C = vectorAdd(scalarMultiply(others[2], BOUNDING_BOX_SCALING_FACTOR), (WIDTH/2, HEIGHT/2))
    D = vectorAdd(scalarMultiply(others[1], BOUNDING_BOX_SCALING_FACTOR), (WIDTH/2, HEIGHT/2))

    Xs = []
    Ys = []
    for i in [A, B, C, D]:
        Xs.append(i[0])
        Ys.append(i[1])
    
    clearanceX = int(ROBOT_BOUNDING_BOX_SIZE // cWidth)
    clearanceY = int(ROBOT_BOUNDING_BOX_SIZE // cHeight)

    Imin = int(min(Ys) // cHeight) - clearanceY
    Imax = int(max(Ys) // cHeight) + clearanceY
    Jmin = int(min(Xs) // cWidth) - clearanceX
    Jmax = int(max(Xs) // cWidth) + clearanceX

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

    def increaseInclusionRange(r, clearance):
        valMin = min(r)
        valMax = max(r)

        lhs = list(range(valMax + 1, valMax + 1 + clearance))
        rhs = list(range(valMin - clearance, valMin))

        return lhs + r + rhs

    for I in range(Imin, Imax + 1):
        for J in range(Jmin, Jmax + 1):
            if I < 0 or J < 0:
                continue
            if I >= len(Nodes) or J >= len(Nodes[0]):
                continue

            if Nodes[J][I].state == 1:
                continue
            # Nodes[J][I].setState(2)
            x, y = dotProduct((J + 0.5, I + 0.5), (cWidth, cHeight))

            ### Marking points inside the bounding box
            # reduces checks for points that are "more" inside
            # ie, greater overlap
            if isPointInsideBoundingBox(x, y):
                Nodes[J][I].setState(1)
                continue

            ### Marking points along the bounding box edges
            # if that comes out as false,
            # see if any of the corners of the cell
            # is inside the bounding box
            atLeastOneInside = False
            for i in increaseInclusionRange([0, 1], clearanceY):
                if atLeastOneInside:
                    break
                for j in increaseInclusionRange([0, 1], clearanceX):
                    x, y = dotProduct((J + j, I + i), (cWidth, cHeight))
                    atLeastOneInside = isPointInsideBoundingBox(x, y)
                    if atLeastOneInside:
                        break
            if atLeastOneInside:
                Nodes[J][I].setState(1)

    # 3. create clearance around the walls
    for i in range(clearanceY):
        for j in range(COLS):
            Nodes[j][i].setState(1)
            Nodes[COLS - j - 1][ROWS - i - 1].setState(1)
    
    for i in range(ROWS):
        for j in range(clearanceX):
            Nodes[j][i].setState(1)
            Nodes[COLS - j - 1][ROWS - i - 1].setState(1)

def draw():
    global obstacles_visible
    for i in obstacles_static:
        mutateNodesUsingBoundingBox(bbox_all[i])

    # for i in obstacles_visible:
    #     mutateNodesUsingBoundingBox(getCorrectBBox(i))

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
    print("Searching path...")
    print(start.state, end.state)
    if start.state == 1 or end.state == 1:
        return []
    PATH = []
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
                PATH.insert(0, tuple(map(lambda t: t + 0.5, x.pos[::-1])))
                x = came_from[x]
            
            end.setState(3)
            return PATH

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
    
    return []

def setup():
    # temp
    global START_NODE, END_NODE
    global ROBOT_BOUNDING_BOX_SIZE
    global robot_pos, target_pos

    startX, startY, startZ = robot_pos
    endX, endY, endZ = target_pos

    cWidth = WIDTH / COLS / BOUNDING_BOX_SCALING_FACTOR
    cHeight = HEIGHT / ROWS / BOUNDING_BOX_SCALING_FACTOR

    startX, startY = vectorAdd((startX, startY), (WIDTH/2/BOUNDING_BOX_SCALING_FACTOR, HEIGHT/2/BOUNDING_BOX_SCALING_FACTOR))
    endX, endY = vectorAdd((endX, endY), (WIDTH/2/BOUNDING_BOX_SCALING_FACTOR, HEIGHT/2/BOUNDING_BOX_SCALING_FACTOR))

    startX = int(startX // cWidth)
    startY = int(startY // cHeight)
    endX = int(endX // cWidth)
    endY = int(endY // cHeight)

    # endX = startX + 2
    # endY = startY
    
    START_NODE = Nodes[startX][startY]
    END_NODE = Nodes[endX][endY]
    START_NODE.setState(2)
    END_NODE.setState(3)

    # Make node connections
    for i in range(ROWS):
        for j in range(COLS):
            if j < COLS - 1:
                Nodes[i][j].connect(Nodes[i][j + 1])
            if i < ROWS - 1:
                Nodes[i][j].connect(Nodes[i+1][j])

    # getting size of the robot
    distances = list(map(lambda x: distance(x, bbox_robot[0]), bbox_robot))
    diagonal = max(distances)

    ROBOT_BOUNDING_BOX_SIZE = diagonal/2 * BOUNDING_BOX_SCALING_FACTOR

def displayPath(path):
    for bbox in bbox_all:
        mutateNodesUsingBoundingBox(bbox)

    drawNodes()
    drawBoundingBoxes()
    drawGrid()

    if path != -1:
        for i in range(len(path) - 1):
            pygame.draw.circle(window, DARKGREY, dotProduct(path[i], (WIDTH/COLS, HEIGHT/ROWS)), 3, 0)
            pygame.draw.line(window, DARKGREY, dotProduct(path[i], (WIDTH/COLS, HEIGHT/ROWS)), dotProduct(path[i+1], (WIDTH/COLS, HEIGHT/ROWS)), 1)
        pygame.draw.circle(window, DARKGREY, dotProduct(path[-1], (WIDTH/COLS, HEIGHT/ROWS)), 3, 0)

    pygame.display.update()

def loop():
    global START_SEARCH
    global START_NODE, END_NODE

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

        _endX, _endY = END_NODE.pos[::-1]
        count = 0
        direction = random.choice([1, 2, 3, 4])
        def isPosValid(x, y):
            return (0 <= x < len(Nodes) and 0 <= y < len(Nodes[0]))
        while count < 1000:
            if direction == 1 and isPosValid(_endX, _endY - 1):
                if Nodes[_endX][_endY - 1].state == 1:
                    direction = random.choice([2, 4])
                    continue

                _endY -= 1
            elif direction == 2 and isPosValid(_endX + 1, _endY):
                if Nodes[_endX + 1][_endY].state == 1:
                    direction = random.choice([1, 3])
                    continue

                _endX += 1
            elif direction == 3 and isPosValid(_endX, _endY + 1):
                if Nodes[_endX][_endY + 1].state == 1:
                    direction = random.choice([2, 4])
                    continue

                _endY += 1
            elif direction == 4 and isPosValid(_endX - 1, _endY):
                if Nodes[_endX - 1][_endY].state == 1:
                    direction = random.choice([1, 3])
                    continue

                _endX -= 1

            Nodes[_endX][_endY].setState(4)
            count += 1

        END_NODE.setState(0)
        END_NODE = Nodes[_endX][_endY]
        END_NODE.setState(3)

        pathFound = a_star(START_NODE, END_NODE)
        if pathFound:
            print(pathFound)
            displayPath(pathFound)
            
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