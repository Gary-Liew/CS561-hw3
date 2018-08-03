import copy
import operator
import sys
import math
actions = walkNorth, walkSouth, walkWest, walkEast, runNorth, runSouth, runWest, runEast = [(1, 0), (-1, 0), (0, -1), (0, 1), (2, 0), (-2, 0), (0, -2), (0, 2)]
walkActions = walkRight, walkUp, walkLeft, walkDown = [(0, 1), (1, 0), (0, -1), (-1, 0)]
runActions = runRight, runUp, runLeft, runDown = [(0, 2), (2, 0), (0, -2), (-2, 0)]
turns = Left, Right = (+1, -1)

def turn_heading(heading, inc, headings):
    return headings[(headings.index(heading) + inc) % len(headings)]

def turn_right(action):
    if action in walkActions:
        return turn_heading(action, Right, walkActions)
    elif action in runActions:
        return turn_heading(action, Right, runActions)

def turn_left(action):
    if action in walkActions:
        return turn_heading(action, Left, walkActions)
    elif action in runActions:
        return turn_heading(action, Left, runActions)

def vector_add(a, b):
    return tuple(map(operator.add, a, b))


def value_iteration(mdp, epsilon):
    #Solve MDP by value iteration
    U1 = {}
    A = {}
    gamma = mdp.gamma
    for s in mdp.states:
        U1[s] = 0
        A[s] = ()
    while True:
        U = copy.deepcopy(U1)
        delta = 0.0
        for s in mdp.states:
            part = float('-inf')
            for a in mdp.actions(s):
                part1 = mdp.R(s, a)+gamma * sum(p * U[s1] for (p, s1) in mdp.T(s, a))
                if(part1>part):
                    A[s] = a
                    part = part1
            U1[s] = part
            delta = max(delta, abs(U1[s] - U[s]))
        if delta < epsilon:
            return A

"""def expected_utility(a, s, U, mdp):
    #the expected utility of doing a in state s.
    return sum(p * U[s1] for (p, s1) in mdp.T(s, a))"""

"""def policy_iteration(mdp):
    U = {s:0 for s in mdp.states}
    pi = {s: random.choice(mdp.actions(s)) for s in mdp.states}
    while True:
        U = policy_evaluation(pi, U, mdp)
        unchanged = True
        for s in mdp.states:
            a = max(mdp.actions(s), key = lambda a: expected_utility(a, s, U, mdp))
            if a != pi[s]:
                pi[s] = a
                unchanged = False
        if unchanged:
            return pi"""

"""def policy_evaluation(pi, U, mdp, k=20):
    R,T,gamma = mdp.R, mdp.T, mdp.gamma
    for i in range(k):
        for s in mdp.states:
            U[s] = R(s, pi[s]) + gamma * sum(p * U[s1] for (p, s1) in T(s,pi[s]))
    return U"""

"""def best_policy(mdp,U):
    #mapping from state to action.
    pi = {}
    for s in mdp.states:
        pi[s] = max(mdp.actions(s), key=lambda a:expected_utility(a, s, U, mdp))
    return pi"""


class MDP:
    """MDP includes initial state,transition model and reward.The transition model is P(S'|S,a),and I use T(S,a) returns
    a list of (p,S') pairs.For reward, is R(S,a), and returns rwalk or rrun. Also keep track of the terminal states,and
    actions for each state."""
    def __init__(self, actlist, gamma, grid, terminals, pwalk, prun, wallLoc, rwalk, rrun):
        self.init = (0, 0)
        self.actlist = actlist
        if not(0 < gamma <= 1):
            raise ValueError("MDP must have gamma value less than 1 and greater than 0")
        self.gamma = gamma
        self.grid = grid
        self.terminals = terminals
        states = set()
        for x in range(len(grid)):
            for y in range(len(grid[0])):
                if grid[x][y]!= None:
                    states.add((x, y))
        self.states = states
        self.pwalk = pwalk
        self.prun = prun
        self.wallLoc = wallLoc
        transitions = {}
        for s in states:
            transitions[s] = {}
            if s not in terminals:
                for a in actlist:
                    transitions[s][a] = self.calculate_T(s, a, pwalk, prun, states)
            else:
                action = (0, 0)
                transitions[s][action] = [(0.0, s)]
        self.transitions = transitions
        if not self.transitions:
            print("Transition table is empty")
        reward = {}
        for s in states:
            reward[s] = {}
            if s not in terminals:
                for a in actlist:
                    reward[s][a] = self.calculate_R(a, rwalk, rrun)
            else:
                action = (0,0)
                reward[s][action] = grid[s[0]][s[1]]
        self.reward = reward


    def R(self, state, action):
        if not self.reward:
            raise ValueError("Reward model is missing")
        else:
            return self.reward[state][action]

    def T(self, state, action):
        if not self.transitions:
            raise ValueError("Transition model is missing")
        else:
            return self.transitions[state][action]

    def actions(self, state):
        if state in self.terminals:
            return [(0,0)]
        else:
            return self.actlist

    def calculate_R(self, action, rwalk, rrun):
        if action in walkActions:
            return rwalk
        else:
            return rrun

    def calculate_T(self, state, action, pwalk, prun, states):
        if action:
            if action in walkActions:
                self.pcorrect = pwalk
            else:
                self.pcorrect = prun
            self.other = 0.5* (1 - self.pcorrect)
            return [(self.pcorrect, self.go(state, action, states)),
                    (self.other, self.go(state, turn_right(action), states)),
                    (self.other, self.go(state, turn_left(action), states))]
        else:
            return [(0.0,state)]

    def go(self, state, dir, states):
        if dir in walkActions:
            state1 = vector_add(state, dir)
            if state1 in states:
                return state1
            else:
                return state
        else:
            first = dir[0]/2
            second = dir[1]/2
            halfdir = (first, second)
            state2 = vector_add(state, halfdir)
            state3 = vector_add(state, dir)
            if state2 not in states or state3 not in states:
                return state
            else:
                return state3

def changeToAction(res):
    actions = ["Exit","Walk Up", "Walk Down", "Walk Left", "Walk Right", "Run Up", "Run Down", "Run Left", "Run Right"]
    actionsList = [(0,0), (1, 0), (-1, 0), (0, -1), (0, 1), (2, 0), (-2, 0), (0, -2), (0, 2)]
    index = actionsList.index(res)
    return actions[index]

def main():
    with open('input.txt') as f:
        content = f.read().splitlines()
    eachNum = content[0].split(',')
    rowsNum = int(eachNum[0])
    colsNum = int(eachNum[1])
    grid = []
    for i in range(0,rowsNum):
        list = []
        for j in range(0,colsNum):
            list.append(0)
        grid.append(list)
    wallNum = int(content[1])
    wallLoc = []
    for i in range(0,wallNum):
        if ',' in content[2+i]:
            loc = content[2+i].split(',')
            rowWall = int(loc[0])
            colWall = int(loc[1])
            wallLoc.append((rowWall-1,colWall-1))
            grid[rowWall-1][colWall-1] = None
        else:
            raise ValueError("input file format is not correct,pls check the number of wall in input file.")
    terminalNum = int(content[2+wallNum])
    terminals = []
    for i in range(0,terminalNum):
        if content[3+wallNum+i].count(",")==2:
            loc = content[3+wallNum+i].split(',')
            rowTerminal = int(loc[0])
            colTerminal = int(loc[1])
            terminals.append((rowTerminal-1,colTerminal-1))
            rewardTerminal = float(loc[2])
            grid[rowTerminal-1][colTerminal-1] = rewardTerminal
        else:
            raise ValueError("input file format is not correct,pls check the number of terminal  in input file.")
    probability=content[3+wallNum+terminalNum].split(',')
    pwalk = float(probability[0])
    prun = float(probability[1])
    rewards = content[4+wallNum+terminalNum].split(',')
    rwalk = float(rewards[0])
    rrun = float(rewards[1])
    gamma = float(content[5+wallNum+terminalNum])
    actlist = actions
    if pwalk == 1 and prun == 1 and rwalk==rrun and wallNum==0:
        length = len(terminals)
        A = [{} for _ in range(length)]
        U = [{} for _ in range(length)]
        for k in range(len(terminals)):
            goal = terminals[k]
            for i in range(0, rowsNum):
                for j in range(0 ,colsNum):
                    if (i, j) in terminals:
                        A[k][(i,j)] = (0, 0)
                        U[k][(i,j)] = grid[i][j]
                    elif (i, j) in wallLoc:
                        A[k][(i,j)] = None
                        U[k][(i,j)] = float('-inf')
                    else:
                        goalRow = goal[0]
                        goalCol = goal[1]
                        if i< goalRow:
                            rowDiff = goalRow - i
                            if rowDiff % 2 == 1:
                                A[k][(i,j)] = (1,0)
                                colDiff = abs(goalCol - j)
                                num = 0
                                if colDiff % 2 == 1:
                                    num = rowDiff/2 + 1 + colDiff/2 + 1
                                else:
                                    num = rowDiff/2 + 1 + colDiff/2
                                U[k][(i,j)] = (rrun*(1 - math.pow(gamma, num))/(1-gamma))+ math.pow(gamma, num)*grid[goalRow][goalCol]
                            else:
                                if j<= goalCol:
                                    colDiff = goalCol - j
                                    if colDiff % 2 ==1:
                                        A[k][(i, j)] = (0, 1)
                                        num = rowDiff/2 + colDiff/2 +1
                                        U[k][(i, j)] = (rrun * (1 - math.pow(gamma, num)) / (1 - gamma)) + math.pow(gamma, num) * grid[goalRow][goalCol]
                                    else:
                                        A[k][(i, j)] = (2, 0)
                                        num = rowDiff/2+colDiff/2
                                        U[k][(i, j)] = (rrun * (1 - math.pow(gamma, num)) / (1 - gamma)) + math.pow(
                                            gamma, num) * grid[goalRow][goalCol]
                                else:
                                    colDiff = j - goalCol
                                    if colDiff % 2 ==1:
                                        A[k][(i, j)] = (0, -1)
                                        num = rowDiff / 2 + colDiff / 2 + 1
                                        U[k][(i, j)] = (rrun * (1 - math.pow(gamma, num)) / (1 - gamma)) + math.pow(
                                            gamma, num) * grid[goalRow][goalCol]
                                    else:
                                        A[k][(i, j)] = (2, 0)
                                        num = rowDiff / 2 + colDiff / 2
                                        U[k][(i, j)] = (rrun * (1 - math.pow(gamma, num)) / (1 - gamma)) + math.pow(
                                            gamma, num) * grid[goalRow][goalCol]
                        elif i == goalRow:
                            if j< goalCol:
                                colDiff = goalCol - j
                                if colDiff%2==1:
                                    A[k][(i, j)] = (0, 1)
                                    num = colDiff / 2 +1
                                    U[k][(i, j)] = (rrun * (1 - math.pow(gamma, num)) / (1 - gamma)) + math.pow(
                                        gamma, num) * grid[goalRow][goalCol]
                                else:
                                    A[k][(i, j)] = (0, 2)
                                    num = colDiff / 2
                                    U[k][(i, j)] = (rrun * (1 - math.pow(gamma, num)) / (1 - gamma)) + math.pow(
                                        gamma, num) * grid[goalRow][goalCol]
                            elif j>goalCol:
                                colDiff = j - goalCol
                                if colDiff%2==1:
                                    A[k][(i, j)] = (0, -1)
                                    num = colDiff / 2 + 1
                                    U[k][(i, j)] = (rrun * (1 - math.pow(gamma, num)) / (1 - gamma)) + math.pow(
                                        gamma, num) * grid[goalRow][goalCol]
                                else:
                                    A[k][(i, j)] = (0, -2)
                                    num = colDiff / 2
                                    U[k][(i, j)] = (rrun * (1 - math.pow(gamma, num)) / (1 - gamma)) + math.pow(
                                        gamma, num) * grid[goalRow][goalCol]
                        else:
                            rowDiff = i - goalRow
                            if rowDiff % 2 == 1:
                                A[k][(i,j)] = (-1,0)
                                colDiff = abs(goalCol - j)
                                num = 0
                                if colDiff % 2 == 1:
                                    num = rowDiff/2 + 1 + colDiff/2 + 1
                                else:
                                    num = rowDiff/2 + 1 + colDiff/2
                                U[k][(i,j)] = (rrun*(1 - math.pow(gamma, num))/(1-gamma))+ math.pow(gamma, num)*grid[goalRow][goalCol]
                            else:
                                if j<= goalCol:
                                    colDiff = goalCol - j
                                    if colDiff % 2 ==1:
                                        A[k][(i, j)] = (0, 1)
                                        num = rowDiff/2 + colDiff/2 +1
                                        U[k][(i, j)] = (rrun * (1 - math.pow(gamma, num)) / (1 - gamma)) + math.pow(gamma, num) * grid[goalRow][goalCol]
                                    else:
                                        A[k][(i, j)] = (-2, 0)
                                        num = rowDiff/2+colDiff/2
                                        U[k][(i, j)] = (rrun * (1 - math.pow(gamma, num)) / (1 - gamma)) + math.pow(
                                            gamma, num) * grid[goalRow][goalCol]
                                else:
                                    colDiff = j - goalCol
                                    if colDiff % 2 ==1:
                                        A[k][(i, j)] = (0, -1)
                                        num = rowDiff / 2 + colDiff / 2 + 1
                                        U[k][(i, j)] = (rrun * (1 - math.pow(gamma, num)) / (1 - gamma)) + math.pow(
                                            gamma, num) * grid[goalRow][goalCol]
                                    else:
                                        A[k][(i, j)] = (-2, 0)
                                        num = rowDiff / 2 + colDiff / 2
                                        U[k][(i, j)] = (rrun * (1 - math.pow(gamma, num)) / (1 - gamma)) + math.pow(
                                            gamma, num) * grid[goalRow][goalCol]

        res = {}
        for i in range(0, rowsNum):
            for j in range(0, colsNum):
                num = float('-inf')
                for k in range(len(terminals)):
                    if U[k][(i,j)]>num:
                        num = U[k][(i,j)]
                        res[(i,j)] = A[k][(i,j)]
        output = ""
        for i in range(rowsNum, 0, -1):
            for j in range(0,colsNum):
                ans = res[(i-1,j)]
                output = output+ changeToAction(ans)+","
            output = output[:-1] + '\n'
        output = output[:-1]
        outfile = open('output.txt', 'w')
        outfile.write(output)
    else:
        decision_env = MDP(actlist, gamma, grid, terminals, pwalk, prun, wallLoc, rwalk, rrun)
        pi = value_iteration(decision_env, sys.float_info.epsilon)
        output = ""
        for i in range(rowsNum, 0, -1):
            for j in range(0, colsNum):
                if (i-1,j) in decision_env.wallLoc:
                    output = output + "None" + ","
                else:
                    res = pi[(i-1, j)]
                    output = output + changeToAction(res) + ","
            output = output[:-1] + '\n'
        output = output[:-1]
        outfile = open('output.txt', 'w')
        outfile.write(output)

main()