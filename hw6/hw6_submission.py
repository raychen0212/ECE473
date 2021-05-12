
from util import manhattanDistance
from game import Directions
import random, util
import queue

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions():
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
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

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):

    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """

    # BEGIN_YOUR_CODE (our solution is 17 lines of code, but don't worry if you deviate from this)

    def recursive(gameState, index, action):

      stop = Directions.STOP
      if gameState.isWin() == 1 or gameState.isLose()==1:
        output = gameState.getScore()
        return (output, stop)
      if gameState.getLegalActions(self.index) == list():
        temp_state = 1
      else:
        temp_state = 0

      if temp_state == 1 or action == 0:
        temp_result = (self.evaluationFunction(gameState), None)
        return temp_result
      elif index != 0:
        v = (float('inf'), stop)
        gaming = gameState.getLegalActions(index)
        for i in gaming:
          num_agent = gameState.getNumAgents() - 1
          if i == stop: continue
          if index == num_agent:
            v = min(v, (recursive(gameState.generateSuccessor(index, i), 0, action - 1)[0], i))
          else:
            v = min(v, (recursive(gameState.generateSuccessor(index, i), index + 1, action)[0], i))
        return v
      elif index == 0:
        v = (float('-inf'), stop)
        gaming = gameState.getLegalActions(index)
        for i in gaming:
            v = max(v, (recursive(gameState.generateSuccessor(index, i), index + 1, action)[0], i))
        return v
    result = recursive(gameState, self.index, self.depth)[1]
    return result
    # END_YOUR_CODE
######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_CODE (our solution is 32 lines of code, but don't worry if you deviate from this)
    def recursive(gameState, depth, index, alpha, beta):
      stop = Directions.STOP
      g_state = gameState.getNumAgents()-1
      if gameState.getLegalActions(self.index) == list():
        temp_state = 1
      else:
        temp_state = 0
      if gameState.isWin() == 1 or gameState.isLose()==1:
        output = gameState.getScore()
        return (output, stop)
      if temp_state == 1 or depth == 0:
        x = self.evaluationFunction(gameState)
        temp_result = (x, stop)
        return temp_result
      elif index != 0:
        v = (float('inf'), stop)
        gaming = gameState.getLegalActions(index)
        for i in gaming:

          if i == stop: continue
          if index == g_state:
            v = min(v, (recursive(gameState.generateSuccessor(index, i), depth - 1, 0, alpha, beta)[0], i))
          else:
            v = min(v, (recursive(gameState.generateSuccessor(index, i), depth, index + 1, alpha, beta)[0], i))
          exact_val = v[0]
          if alpha <= exact_val:
            beta = min(beta, exact_val)
          else:
            return v
        return v
      elif index == 0:
        v = (float('-inf'), stop)
        gaming = gameState.getLegalActions(index)
        for i in gaming:
          if i == Directions.STOP:continue
          v = max(v, (recursive(gameState.generateSuccessor(index, i), depth, index + 1, alpha, beta)[0], i))
          exact_val = v[0]
          if exact_val <= beta:
            alpha = max(alpha, exact_val)
          else:
            return v
        return v
    action = recursive(gameState, self.depth, self.index, float('-inf'), float('inf'))[1]
    return action

    # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_CODE (our solution is 17 lines of code, but don't worry if you deviate from this)

    def recursive(gameState, depth, index):
      stop = Directions.STOP
      g_state = gameState.getNumAgents()-1
      if gameState.getLegalActions(self.index) == list():
        temp_state = 1
      else:
        temp_state = 0
      if gameState.isWin() == 1 or gameState.isLose()==1:
        output = gameState.getScore()
        return (output, stop)
      if temp_state == 1 or depth == 0:
        x = self.evaluationFunction(gameState)
        temp_result = (x, stop)
        return temp_result
      elif index != 0:
        temp_list = []
        gaming = gameState.getLegalActions(index)
        for i in gaming:
          if i == stop: continue
          if index == g_state:
            calc = recursive(gameState.generateSuccessor(index, i), depth - 1, 0)[0]
          else:
            calc = recursive(gameState.generateSuccessor(index, i), depth, index + 1)[0]
          temp_list.append(calc)
        average = sum(temp_list) / len(temp_list)
        return (average, i)
      elif index == 0:
        v = (float('-inf'), stop)
        gaming = gameState.getLegalActions(index)
        for i in gaming:
          if i == Directions.STOP:continue
          v = max(v, (recursive(gameState.generateSuccessor(index, i), depth, index + 1)[0], i))
        return v
    action = recursive(gameState, self.depth, self.index)[1]
    return action

    # END_YOUR_CODE

######################################################################################
# Problem 4a : creating a better evaluation function

def betterEvaluationFunction(currentGameState):
    """
      Your extreme, unstoppable evaluation function (problem 4).

      DESCRIPTION: <write something here so we know what you did>
    """

    # BEGIN_YOUR_CODE (our solution is 17 lines of code, but don't worry if you deviate from this)
    position = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghost = currentGameState.getGhostStates()
    score = currentGameState.getScore()
    caps = currentGameState.getCapsules()
    food_list = []
    cap_list = []
    minimum_food = 0
    cap_dist = 0
    score_cap = 0
    len_dist = float("inf")
    for i in food.asList():
      if manhattanDistance(position, i) < len_dist:
        food_list.append(manhattanDistance(position, i))
    for i in caps:
      dist = util.manhattanDistance(position, i)
      if dist < len_dist:
        cap_dist = dist
    if cap_dist > 0:
      minimum_food = 1.0 / min(food_list)
    for capsule in caps:
      cap_list.append(1.0 / manhattanDistance(position, capsule))
      score_cap = len(cap_list)
    if food_list:
      sum_of_food = sum(food_list)
      len_of_food = len(food_list)
      score = score + (1.0 / (sum_of_food / len_of_food)) + minimum_food
    for current_ghost in ghost:
      if current_ghost.scaredTimer < manhattanDistance(position, current_ghost.getPosition()):
        if current_ghost.scaredTimer <= 0:
          score = score - min(manhattanDistance(position, current_ghost.getPosition()), float("inf"))
        else:
          score = score + min(manhattanDistance(position, current_ghost.getPosition()), float("inf"))
      elif current_ghost.scaredTimer >= manhattanDistance(position, current_ghost.getPosition()):
        score = score + 100

    score += score_cap
    return score

    # END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction

# Problem 4b : Describe your evaluation function.
# We first need to get the position of the Pacman, food, the ghosts, capsule and the original score.
# There are two part that will be deciding the final score, when the ghosts are scared and when it is a normal ghost.
# Next, we need to find the capsule score and whenever pacman receive the capsule it will go back and eat the ghost.
# Next, we will need to find the average distance to the food and take the reciprocal of it.
# We also need to find the reciprocal of the minimum food distance.
# If the ghost are scared, we will find the minimum distance of the manhattan distance and subtract the value.
# If the ghost are not scared, we then add the minimum manhattan distance.
# Eventually, we will add the score for eating the capsule back to the Pacman.
