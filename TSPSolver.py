#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools

NUM_GREEDY_TRIES_BB = 5

class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

	class cityData:
		def __init__(self, prev, cit):
			self.previous = prev
			self.city = cit


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	startingCity = None
	finalCity = None
	def greedy(self, time_allowance=60.0):
		start_time = time.time()
		# initialize unvisited[] to all cities
		unvisited = self._scenario.getCities().copy()  # would it be better to just have an array of
		# sizeof(unvisited) and initialize all values to 0?
		# previousPointers = []
		# for c in unvisited:
		# 	previousPointers.append(self.cityData(None, c))

		startingCityIndex = 0
		startingCity = unvisited[startingCityIndex]

		route = []
		currentCity = startingCity
		while len(unvisited) != 0:
			unvisited.remove(currentCity)
			route.append(currentCity)
			# find shortest path to city2 from current city1
			nextCity = self.findClosestCity(currentCity, unvisited)

			# update previous pointers
			# previousPointers[nextCity._index].previous = currentCity
			if len(unvisited) == 0:
				if self.checkLastCity(startingCity, currentCity) == False:
					startingCityIndex += 1
					unvisited = self._scenario.getCities().copy()
					route.clear()
					currentCity = unvisited[startingCityIndex]
					nextCity = currentCity
					continue
				else: break

			if nextCity == currentCity:
				startingCityIndex += 1
				unvisited = self._scenario.getCities().copy()
				route.clear()
				currentCity = unvisited[startingCityIndex]

			currentCity = nextCity

		results = {}
		foundTour = False
		bssf = TSPSolution(route)
		count = len(route)
		if bssf.cost < np.inf:
			# Found a valid route
			foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else 999999
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None

		return results

	def findClosestCity(self, currentCity, unvisited):

		# loop through all edges of city1
		edges = self._scenario.getEdges()[currentCity._index]
		cities = self._scenario.getCities()
		shortestDistance = 999999
		closestCity = currentCity
		for i in range(len(edges)):
			if edges[i] and (cities[i] in unvisited ):
				distance = self.calculateDistance(cities[i], currentCity)
				if distance < shortestDistance:
					shortestDistance = distance
					closestCity = cities[i]

		return closestCity


	def checkLastCity(self, firstCity, currentCity):
		edges = self._scenario.getEdges()[currentCity._index]
		if edges[firstCity._index] == True:
			return True

		return False


	def calculateDistance(self, city1, city2):
		y = city1._y - city2._y
		x = city1._x - city2._x

		dist = np.sqrt(y**2 + x**2)

		return dist

	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''

	#############
	# core functions
	#############
	def branchAndBound( self, time_allowance=60.0 ):
		start_time = time.time()
		self.cities = self._scenario.getCities()
		self.pq = []
		self.depthPriority = 10
		self.numNodes = len(self.cities)
		#init adjacency matrix
		self.matrix = self.makeMatrixFromEdgelist()

		self.lowerBound, self.lowerBoundMatrix = self.calculateLowerBound()
		self.upperBound = self.calculateUpperBound()
		startingState = BBState(0, self.lowerBoundMatrix, 0)
		self.doBranch(startingState)

	def calculateLowerBound(self):
		return self.normalizeMatrixInit(self.matrix)

	def calculateUpperBound(self):
		bestResult = self.greedy(60.0)
		for i in range(NUM_GREEDY_TRIES_BB):
			result = self.greedy(60.0)
			if result['cost'] < bestResult['cost']:
				bestResult = result

		cost = 0
		previousCity = None
		for currentCity in bestResult['soln'].route:
			if previousCity != None:
				cost += self.calculateDistance(currentCity, previousCity)

			previousCity = currentCity

		return cost

	def doBranch(self, BBstate):
		for i in range(len(self.matrix[BBState.nodeNumber])):
			if self.matrix[BBState.nodeNumber][i] == True:
				newState = self.calculateEdge(BBState, i)
				if newState.cost < self.upperBound:
					heapq.heappush(self.pq, (self.calculatePriority(newState.level, newState.cost), newState))






	#############
	# aux functions
	#############
	def calculateEdge(self, BBState, nextStateInt):
		newMatrix = BBState.matrix
		for i in len(newMatrix):
			newMatrix[i][BBState.nodeNumber] = math.inf
		for i in len(newMatrix):
			newMatrix[nextStateInt][i] = math.inf

		cost, newMatrix = self.normalizeMatrix(newMatrix, BBState.nodeNumber, nextStateInt)
		newState = BBState(BBState.level + 1, newMatrix, nextStateInt)
		newState.path.append(BBState.nodeNumber)
		newState.cost = cost
		return newState

	#returns cost, newMatrix
	def normalizeMatrix(self, matrix, clearedRow, clearedColumn):
		rows = len(matrix)
		columns = len(matrix[0])
		cost = 0
		newMatrix = matrix

		for i in range(rows):
			if i == clearedRow:
				smallestVal = math.inf
				for j in range(columns):
					if newMatrix[i][j] < smallestVal:
						smallestVal = newMatrix[i][j]
				for j in range(columns):
					newMatrix[i][j] = newMatrix[i][j] - smallestVal
				cost += smallestVal

		for i in range(columns):
			if i == clearedColumn:
				smallestVal = math.inf
				for j in range(rows):
					if newMatrix[i][j] < smallestVal:
						smallestVal = newMatrix[i][j]
				for j in range(rows):
					newMatrix[i][j] = newMatrix[i][j] - smallestVal
				cost += smallestVal


		return cost, newMatrix

	def normalizeMatrixInit(self, matrix):
		rows = len(matrix)
		columns = len(matrix[0])
		cost = 0
		newMatrix = matrix

		for i in range(rows):
			smallestVal = math.inf
			for j in range(columns):
				if newMatrix[i][j] < smallestVal:
					smallestVal = newMatrix[i][j]
			for j in range(columns):
				newMatrix[i][j] = newMatrix[i][j] - smallestVal
			cost += smallestVal

		for i in range(columns):
			smallestVal = math.inf
			for j in range(rows):
				if newMatrix[i][j] < smallestVal:
					smallestVal = newMatrix[i][j]
			for j in range(rows):
				newMatrix[i][j] = newMatrix[i][j] - smallestVal
			cost += smallestVal


		return cost, newMatrix

	def makeMatrixFromEdgelist(self):
		edges = self._scenario.getEdges()
		cities = self._scenario.getCities()
		matrix = np.array([[math.inf for i in range(len(edges))] for i in range(len(edges[0]))])

		for i in range(len(edges)):
			for j in range(len(edges[0])):
				if edges[i][j] == True:
					matrix[i][j] = self.calculateDistance(cities[i], cities[j])

		return matrix

	#calculates priority score
	def calculatePriority(self, level, cost):
		costRatio = self.upperBound - cost/self.upperBound
		levelRatio = level * self.depthPriority / self.numNodes
		return (costRatio + levelRatio) * 100


	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy( self,time_allowance=60.0 ):
		pass
		



class BBState:
	def __init__(self, level, matrix, nodeNumber):
		self.level = level
		self.matrix = matrix
		self.nodeNumber = nodeNumber
		self.cost = 0
		self.path = []


