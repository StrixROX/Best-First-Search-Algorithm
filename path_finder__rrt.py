# RRT Based Path Finding Algorithm Visualisation

import pygame
import random
import time

# Initialize Pygame
pygame.init()

# Define the width and height of the window
WIDTH, HEIGHT = (800, 800)

# Define the number of rows and columns in the grid
ROWS, COLS = (40, 40)

# Calculate the width and height of each grid cell
CWIDTH = WIDTH // COLS
CHEIGHT = HEIGHT // ROWS

# Create the game window
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))

# Set the title of the window
pygame.display.set_caption("RRT Based Path Finding Algorithm Visualisation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
DARKGREY = (150, 150, 150)
GREEN = (20, 217, 72)
RED = (255, 0, 0)
TURQUOISE = (64, 224, 208)
PURPLE = (160, 32, 240)

# Misc. Properties
CLICK_MODE = 0
# 0: creating / clearing obstacles
# 1: creating / clearing start and goal nodes
START_SEARCH = 0
MAXLEN = 3 # max distance of new node from nearest node
SEARCHDIST = 3 # max search distance for nodes with shorted path (for RRT*)

SHOWINGPATH = 0 # pauses program when solution path is found

START_NODE = None
END_NODE = None

class GridNode:
	def __init__(self, pos, state = 0):
		self.pos = pos
		self.x, self.y = self.pos
		self.id = pos[1]*COLS + pos[0]+1
		self.state = state
		self.colorMap = {
			0: WHITE,			# 0: empty, traversable node
			1: BLACK,			# 1: blocked, non-traversable node
			2: GREEN,			# 2: start node
			3: RED,				# 3: goal node
			4: DARKGREY,	# 4: searching node for path
			5: TURQUOISE,	# 5: searching node neighbour for path
			6: PURPLE,		# 6: path found
		}

	def setState(self, to):
		self.state = to

	def getColor(self):
		return self.colorMap[self.state]

	def __hash__(self):
		return hash((self.id, self.state, self.x, self.y))

	def __eq__(self, __value):
		if not isinstance(__value, Node):
				return False

		return self.pos == __value.pos

	def __str__(self):
		# return self.id + repr(self.pos)
		return str(self.id)# + repr(tuple(self.neighbours))

	def __repr__(self):
		return str(self.id)
		# return self.id + repr(tuple(self.neighbours))

	def show(self):
		x, y = (self.x * CWIDTH, self.y * CHEIGHT)
		pygame.draw.rect(WINDOW, self.colorMap[self.state], (x, y, x + CWIDTH, y + CHEIGHT))
GRID = [[GridNode((j, i), 0) for j in range(COLS)] for i in range(ROWS)]

class Node(GridNode):
	def __init__(self, pos, _from = None, _cost = 0):
		super().__init__(pos, state = 0)
		self.prev = _from
		self.cost = _cost

	def __eq__(self, __value):
		if not isinstance(__value, Node):
				return False

		return self.pos == __value.pos

def euclideanDistance(node_a, node_b):
	x1, y1 = node_a.pos
	x2, y2 = node_b.pos

	return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

class RRT:
	def __init__(self, grid):
		self.grid = grid
		self.nodeList = self.__getNodeList()
		self.obstacles = self.__getObstacles()
		self.traversable = self.__getTraversable()
		self.tree = [Node(START_NODE.pos)]

	def __getNodeList(self):
		nodeList = []
		for i in self.grid:
			nodeList += i

		return nodeList

	def __getObstacles(self):
		obstacles = []
		for i in self.nodeList:
			if i.state == 1:
				obstacles.append(i)

		return obstacles

	def __getTraversable(self):
		traversable = []
		for i in self.nodeList:
			if i.state != 1:
				traversable.append(i)

		return traversable

	def pathIsBlocked(self, a, b):
		x1, y1 = a.pos
		x2, y2 = b.pos
		
		for i in range(y1, y2 + 1):
			for j in range(x1, x2 + 1):
				if self.grid[i][j].state == 1:
					return True

		# distAB = euclideanDistance(a, b)
		# for i in self.obstacles:
		# 	distAI = euclideanDistance(a, i)
		# 	distBI = euclideanDistance(b, i)

		# 	if distAI + distBI == distAB:
		# 		return True

		return False

	def findNearestNode(self, a):
		nodes = sorted(self.tree, key=lambda x: euclideanDistance(x, a))

		return (nodes[0], euclideanDistance(nodes[0], a))

	def createRandNode(self):
		while True:
			randGridNode = random.choice(self.traversable)
			randNode = Node(randGridNode.pos)
			nearestNode, distance = self.findNearestNode(randNode)

			if self.pathIsBlocked(randNode, nearestNode):
				print(randNode.pos, nearestNode.pos)
				continue

			if distance <= MAXLEN:
				# nearestNode.next = randNode
				randNode.prev = nearestNode
				randNode.cost = nearestNode.cost + distance

				return (randNode, nearestNode)
			else:
				x1, y1 = nearestNode.pos
				x2, y2 = randNode.pos

				if x2 != x1:
					tan = (y2 - y1) / (x2 - x1)

					x = x1 + MAXLEN / ((1 + tan**2)**0.5) * (1 if x1 < x2 else -1)
					y = y1 + MAXLEN / ((1 + tan**2)**0.5) * (1 if y1 < y2 else -1) * tan
				else:
					x = x1
					y = y1 + MAXLEN * (1 if y1 < y2 else -1)


				newNode = Node((int(x), int(y)))
				newNode.prev = nearestNode
				newNode.cost = nearestNode.cost + MAXLEN

				if newNode in self.obstacles:
					continue
				# nearestNode.next = newNode

				return (newNode, nearestNode)

	def drawLine(self, p1, p2, color = TURQUOISE):
		if p1 is None or p2 is None:
			return
		x1, y1 = p1.pos
		x2, y2 = p2.pos
		pygame.draw.line(WINDOW, color, ((x1 + 0.5) * CWIDTH, (y1 + 0.5) * CHEIGHT), ((x2 + 0.5) * CWIDTH, (y2 + 0.5) * CHEIGHT), 3)

	def drawPoint(self, p, color = TURQUOISE):
		x, y = p.pos
		pygame.draw.circle(WINDOW, color, ((x + 0.5) * CWIDTH, (y + 0.5) * CHEIGHT), 5, 0)

	def tracePath(self):
		nearestNodeToEnd, _ = self.findNearestNode(END_NODE)

		self.drawLine(END_NODE, nearestNodeToEnd, color=GREEN)
		
		node = nearestNodeToEnd

		pygame.display.update()

		while node is not None:
			self.drawLine(node, node.prev, color=GREEN)
			node = node.prev

			pygame.display.update()

	def run(self):
		while self.findNearestNode(END_NODE)[1] > MAXLEN:
			randNode, nearestToRandNode = self.createRandNode()

			self.drawPoint(randNode)
			self.drawLine(randNode, nearestToRandNode)
			pygame.display.update()

			self.tree.append(randNode)

		self.tracePath()

def drawGrid():
	# Draw the grid
	for i in range(ROWS):
		pygame.draw.line(WINDOW, GREY, (0, i * CHEIGHT), (WIDTH, i * CHEIGHT))

	for j in range(COLS):
		pygame.draw.line(WINDOW, GREY, (j * CWIDTH, 0), (j * CWIDTH, HEIGHT))

def draw():
	global GRID
	# Clear the window
	WINDOW.fill((255, 255, 255))

	# Draw grid cells
	for i in range(ROWS):
		for j in range(COLS):
			GRID[i][j].show()

	# Draw the grid
	drawGrid()

	# Update the display
	pygame.display.update()

def handleClicks():
	global START_NODE, END_NODE

	if pygame.mouse.get_pressed()[0]: # LEFT
		x, y = pygame.mouse.get_pos()

		i = min(max(int(y // CHEIGHT), 0), ROWS - 1)
		j = min(max(int(x // CWIDTH), 0), COLS - 1)

		gridNode = GRID[i][j]

		if CLICK_MODE == 0:
			if gridNode.state != 1:
				gridNode.setState(1)
		elif CLICK_MODE == 1:
			if gridNode.state != 2:
				if START_NODE is not None:
					START_NODE.setState(0)
				START_NODE = gridNode
				START_NODE.setState(2)
	elif pygame.mouse.get_pressed()[2]: # RIGHT
		x, y = pygame.mouse.get_pos()

		i = min(max(int(y // CHEIGHT), 0), ROWS - 1)
		j = min(max(int(x // CWIDTH), 0), COLS - 1)

		gridNode = GRID[i][j]

		if CLICK_MODE == 0:
			if gridNode.state != 0:
				gridNode.setState(0)
		elif CLICK_MODE == 1:
			if gridNode.state != 3:
				if END_NODE is not None:
					END_NODE.setState(0)
				END_NODE = gridNode
				END_NODE.setState(3)

def setup():
	pass

def loop():
	global START_SEARCH, SHOWINGPATH
	if not SHOWINGPATH:
		handleClicks()
		draw()

		if START_SEARCH == 1:
			rrt = RRT(GRID)
			rrt.run()
			SHOWINGPATH = 1
			START_SEARCH = 0

def GameLoop():
	global CLICK_MODE, START_SEARCH
	setup()

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

	# Quit the program
	pygame.quit()
GameLoop()