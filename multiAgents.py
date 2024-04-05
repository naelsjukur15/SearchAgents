# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodList = newFood.asList()
        nextClosestFood = float('inf')
        for food in foodList:
            nextClosestFood = min(nextClosestFood, manhattanDistance(food, newPos))

        if successorGameState.isWin():
            return float('inf')

        if successorGameState.isLose():
            return float('-inf')

        nextGameScore = successorGameState.getScore()

        return nextGameScore + 1 / nextClosestFood


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """

    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        possibleActions = gameState.getLegalActions(0)
        maximumValue = float('-inf')
        returnAction = 0

        for action in possibleActions:
            successor = gameState.generateSuccessor(0, action)
            value = self.miniMax(successor, 1, 0)
            if value > maximumValue:
                maximumValue = value
                returnAction = action

        return returnAction

    def miniMax(self, gameState: GameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        nextAgent = (agentIndex + 1) % gameState.getNumAgents()

        if nextAgent == 0:
            successorDepth = depth + 1
        else:
            successorDepth = depth

        if agentIndex == 0:
            v = float('-inf')
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                v = max(v, self.miniMax(successor, nextAgent, successorDepth))
        else:
            v = float('inf')
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                v = min(v, self.miniMax(successor, nextAgent, successorDepth))

        return v




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        possibleActions = gameState.getLegalActions(0)
        maximumValue = float('-inf')
        returnAction = 0
        alpha = float('-inf')
        beta = float('inf')

        for action in possibleActions:
            successor = gameState.generateSuccessor(0, action)
            value = self.alphaBeta(successor, 1, 0, alpha, beta)
            if value > maximumValue:
                maximumValue = value
                returnAction = action
            alpha = max(alpha, maximumValue)

        return returnAction

    def alphaBeta(self, gameState: GameState, agentIndex, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        nextAgent = (agentIndex + 1) % gameState.getNumAgents()

        if nextAgent == 0:
            successorDepth = depth + 1
        else:
            successorDepth = depth

        if agentIndex == 0:
            v = float('-inf')
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                v = max(v, self.alphaBeta(successor, nextAgent, successorDepth, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
        else:
            v = float('inf')
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                v = min(v, self.alphaBeta(successor, nextAgent, successorDepth, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)

        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        possibleActions = gameState.getLegalActions(0)
        maximumValue = float('-inf')
        returnAction = 0

        for action in possibleActions:
            successor = gameState.generateSuccessor(0, action)
            value = self.expectiMax(successor, 1, 0)
            if value > maximumValue:
                maximumValue = value
                returnAction = action

        return returnAction

    def expectiMax(self, gameState: GameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        if nextAgent == 0:
            successorDepth = depth + 1
        else:
            successorDepth = depth

        numActions = len(gameState.getLegalActions(agentIndex))

        if agentIndex == 0:
            v = float('-inf')
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                v = max(v, self.expectiMax(successor, nextAgent, successorDepth))
        else:
            v = 0
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                v += self.expectiMax(successor, nextAgent, successorDepth) / numActions  

        return v


def betterEvaluationFunction(state: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """

    "*** YOUR CODE HERE ***"
    currPos = state.getPacmanPosition()
    currFood = state.getFood()
    ghostStates = state.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    foodList = currFood.asList()
    nearestFoodDistance = float('inf')
    for food in foodList:
        nearestFoodDistance = min(nearestFoodDistance, manhattanDistance(food, currPos))

    nearestGhost = float('inf')
    for ghostState in ghostStates:
        nearestGhost = min(nearestGhost, manhattanDistance(currPos, ghostState.getPosition()))

    nearestScaredGhost = float('inf')
    for ghostState, scaredTime in zip(ghostStates, scaredTimes):
        if scaredTime > 0:
            distance = manhattanDistance(currPos, ghostState.getPosition())
            nearestScaredGhost = min(nearestScaredGhost, distance)

    foodLeft = state.getNumFood()

    toReturn = state.getScore()
    toReturn -= 2 * foodLeft
    if nearestFoodDistance != 0:
        toReturn += 2 / nearestFoodDistance
    if nearestGhost < 2:
        toReturn -= 2 / (nearestGhost + 1)
    if nearestScaredGhost != 0:
        toReturn += 1 / (nearestScaredGhost + 1)

    return toReturn

# Abbreviation
better = betterEvaluationFunction
