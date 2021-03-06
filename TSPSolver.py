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
INF = 999999


class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario(self, scenario):
		self._scenario = scenario

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
		foundTour = False
		startingCityIndex = 0
		self.cities = self._scenario.getCities()
		results = {}

		unvisited = self._scenario.getCities().copy()  # would it be better to just have an array of


		startingCity = unvisited[startingCityIndex]

		#keep looping until all nodes are visited, will reset if path isn't complete
		route = []
		currentCity = startingCity
		while len(unvisited) != 0:
			unvisited.remove(currentCity)
			route.append(currentCity)
			# find shortest path to city2 from current city1
			nextCity = self.findClosestCity(currentCity, unvisited)

			if len(unvisited) == 0:
				#reset if failed at end of tour
				if self.checkLastCity(startingCity, currentCity) == False:
					startingCityIndex += 1
					unvisited = self._scenario.getCities().copy()
					route.clear()
					currentCity = unvisited[startingCityIndex]
					nextCity = currentCity
					continue
				else:#break if successful
					break

			#reset if failed attempt midway through tour
			if nextCity == currentCity:
				startingCityIndex += 1
				unvisited = self._scenario.getCities().copy()
				route.clear()
				if startingCityIndex >= len(unvisited): break
				currentCity = unvisited[startingCityIndex]

			currentCity = nextCity

		#if no tour was found return default random tour
		if startingCityIndex == len(self._scenario.getCities()): return self.defaultRandomTour(time_allowance)
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

	#helps find next city to visit using
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
		# y = city1._y - city2._y
		# x = city1._x - city2._x
		#
		# dist = np.sqrt(y**2 + x**2) * 1000

		return city1.costTo(city2)

	
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
		self.start_time = time.time()
		self.cities = self._scenario.getCities()
		self.pq = []
		self.depthPriority = 10
		self.numNodes = len(self.cities)
		self.bestRoutSoFar = []
		self.intermediateSolutions = 0
		self.numStatesMade = 0
		self.maxPq = 0
		self.pruned = 0

		#init adjacency matrix for lower bound
		self.matrix = self.makeMatrixFromEdgelist()
		self.edges = self._scenario.getEdges()
		#make lower bound with above matrix
		self.lowerBound, self.lowerBoundMatrix = self.calculateLowerBound()

		#make upper bound with greedy. I used the one we made in our group project
		self.upperBound = self.calculateUpperBound()
		startingState = BBState(0, self.lowerBoundMatrix, 0)
		startingState.cost = self.lowerBound

		#keep branching on the best state. States are prioritized by 'calculatePriority' function below
		self.doBranch(startingState)
		while time.time() - self.start_time < time_allowance and len(self.pq) > 0:#len(self.pq) > 0:#
			if len(self.pq) > self.maxPq: self.maxPq = len(self.pq)
			self.doBranch(heapq.heappop(self.pq))



		route = []
		for i in self.bestRoutSoFar[0]:
			route.append(self.cities[i])

		results = {}
		foundTour = False
		bssf = TSPSolution(route)
		if bssf.cost < np.inf:
			# Found a valid route
			foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else 999999
		results['time'] = time.time() - self.start_time
		print(self.bestRoutSoFar[2])
		results['count'] = self.bestRoutSoFar[1]
		results['soln'] = bssf
		results['max'] = self.maxPq
		results['total'] = self.numStatesMade
		self.pruned += len(self.pq)
		results['pruned'] = self.pruned

		return results

	#lowerbound function
	def calculateLowerBound(self):
		return self.normalizeMatrixInit(self.matrix)

	#upperbound function, just does greedy
	def calculateUpperBound(self):

		bestResult = self.greedy(60.0)
		# for i in range(NUM_GREEDY_TRIES_BB):
		# 	result = self.greedy(60.0)
		# 	if result['cost'] < bestResult['cost']:
		# 		bestResult = result

		cost = 0
		previousCity = None
		for currentCity in bestResult['soln'].route:
			if previousCity != None:
				cost += self.calculateDistance(currentCity, previousCity)

			previousCity = currentCity

		firstCity = bestResult['soln'].route[0]
		cost += self.calculateDistance(firstCity, previousCity)

		for i in bestResult['soln'].route:
			self.bestRoutSoFar.append(i._index)
		self.bestRoutSoFar = self.bestRoutSoFar, 0, 0
		return cost

	#consider one state, make child states and prune those above upperbound. child states are assign their priority here
	def doBranch(self, state):
		# check if newState is the final state in the route and update upper bound if possible
		if len(state.path) >= self.numNodes - 1:
			if self.edges[state.nodeNumber][state.path[0]] == True:
				newState = self.calculateEdge(state, state.path[0])
				if newState.cost < self.upperBound:
					print(newState.cost)
					print(self.upperBound)
					self.intermediateSolutions += 1
					self.bestRoutSoFar = newState.path, self.intermediateSolutions, time.time() - self.start_time
					self.upperBound = newState.cost
					print("FOUND SOLUTION")
				else:
					self.pruned += 1
		else:
			for i in range(len(self.edges)):
				if self.edges[state.nodeNumber][i] == True:
					if state.path.__contains__(i) == False and len(state.path) < self.numNodes:
						if state.matrix[state.nodeNumber][i] >= 0:
							newState = self.calculateEdge(state, i)
							if newState.cost < self.upperBound:
								newState.priority = self.calculatePriority(newState.level, newState.cost)
								heapq.heappush(self.pq, newState)
							else:
								self.pruned += 1


	#############
	# aux functions
	#############
	#detemines cost of edge and new matrix. returns a the newState
	def calculateEdge(self, state, nextStateInt):
		self.numStatesMade += 1
		newMatrix = state.matrix.copy()


		cost, newMatrix = self.normalizeMatrix(newMatrix, state.nodeNumber, nextStateInt)
		newState = BBState(state.level + 1, newMatrix, nextStateInt)
		newState.path.extend(state.path)
		newState.path.append(state.nodeNumber)
		newState.cost = state.cost + cost
		return newState

	#returns cost, newMatrix
	def normalizeMatrix(self, matrix, startingStateAndZeroRow, newStateAndZerocolumn):
		rows = len(matrix)
		columns = len(matrix[0])
		cost = matrix[startingStateAndZeroRow][newStateAndZerocolumn]
		newMatrix = matrix.copy()


		rowsToCheck = []
		columnsToCheck = []
		#zero out row and column to clear and determine which rows columns need to be normalized
		for i in range(len(newMatrix)):
			if newMatrix[i][newStateAndZerocolumn] >= 0:
				newMatrix[i][newStateAndZerocolumn] = -1
				if rowsToCheck.__contains__(i) == False:
					rowsToCheck.append(i)
		for i in range(len(newMatrix)):
			if newMatrix[startingStateAndZeroRow][i] >= 0:
				newMatrix[startingStateAndZeroRow][i] = -1
				if columnsToCheck.__contains__(i) == False:
					columnsToCheck.append(i)
		rowsToCheck.remove(startingStateAndZeroRow)

		#zero out conjugate edge of one currently considering
		newMatrix[newStateAndZerocolumn][startingStateAndZeroRow] = -1

		#check only the rows and columns that arent' zeroed out
		for i in rowsToCheck:
			smallestVal = INF
			for j in range(columns):
				if newMatrix[i][j] < smallestVal and newMatrix[i][j] >= 0:
					smallestVal = newMatrix[i][j]
			for j in range(columns):
				newMatrix[i][j] = newMatrix[i][j] - smallestVal
			cost += smallestVal

		for j in columnsToCheck:
			smallestVal = INF
			for i in range(rows):
				if newMatrix[i][j] < smallestVal  and newMatrix[i][j] >= 0:
					smallestVal = newMatrix[i][j]
			for i in range(rows):
				newMatrix[i][j] = newMatrix[i][j] - smallestVal
			cost += smallestVal


		return cost, newMatrix

	#used only for calculating lower bound
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
				if newMatrix[j][i] < smallestVal:
					smallestVal = newMatrix[j][i]
			for j in range(rows):
				newMatrix[j][i] = newMatrix[j][i] - smallestVal
			cost += smallestVal


		return cost, newMatrix

	#used for setting up the original matrix. Nonexistant edges left as math.inf
	def makeMatrixFromEdgelist(self):
		edges = self._scenario.getEdges()
		cities = self._scenario.getCities()
		matrix = np.array([[math.inf for i in range(len(edges))] for i in range(len(edges[0]))])

		for i in range(len(edges)):
			for j in range(len(edges[0])):
				if edges[i][j] == True:
					matrix[i][j] = self.calculateDistance(cities[i], cities[j])

		return matrix

	#calculates priority score using the states depth/level and it's cost.
	# favors depth and low cost
	# Also takes into account how many solution have been found.
	# When less have been found level is given more priority
	# a smaller score is higher priority
	def calculatePriority(self, level, cost):

		#ratio for number of solutions so far
		numSolutionsRatio = (self.numNodes - self.intermediateSolutions) / (self.numNodes * level)  #lots of nodes, high for low level
		#numSolutionsRatio = 0

		#cost and levels ratio
		costRatio = cost / self.upperBound
		levelRatio = self.numNodes / (level * self.depthPriority) #level * self.depthPriority / self.numNodes
		return (costRatio + levelRatio + numSolutionsRatio) * 100


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
		self.priority = math.inf

	def __lt__(self, other):
		return self.priority < other.priority



