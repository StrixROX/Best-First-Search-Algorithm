# RRT Based Path Finding Algorithm Visualisation

import pygame

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

START_NODE = None
END_NODE = None

class GridNode:
	def __init__(self, id, pos, state = 0):
		self.id = id
		self.pos = pos
		self.x, self.y = self.pos
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

	def show(self):
		x, y = (self.x * CWIDTH, self.y * CHEIGHT)
		pygame.draw.rect(WINDOW, self.colorMap[self.state], (x, y, x + CWIDTH, y + CHEIGHT))
GRID = [[GridNode(i*COLS + j+1, (j, i), 0) for j in range(COLS)] for i in range(ROWS)]

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

		pygame.draw.rect(WINDOW, self.color_map[self.state], (x, y, WIDTH / COLS, HEIGHT / ROWS), 1)

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
	handleClicks()
	draw()

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