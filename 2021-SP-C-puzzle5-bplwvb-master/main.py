import sys
from math import sqrt
import random
from queue import Queue
from queue import LifoQueue
from queue import PriorityQueue
import copy
import pprint

random.seed()

#used to correctly get the information from the input file and returns all relevent information
def parseInput():
    infile =  open(sys.argv[1])
    mapsize = (infile.readline())
    maprows = mapsize[:mapsize.find(' ')]
    mapcols = mapsize[mapsize.find(' ')+1:mapsize.find('\n')]
    pacmap = [['' for i in range(int(mapcols))] for j in range(int(maprows))]

    maprows = int(maprows)
    mapcols = int(mapcols)


    #takes in map while removing newlines
    temp = ''
    for i in range(maprows):
        temp += (infile.readline().strip())

    #creates the map using input file
    counter = 0
    for i in range(maprows):
        for j in range(mapcols):
            pacmap[i][j] = temp[counter]
            counter += 1


    #initial directions of the ghosts
    initdirections = infile.readline()

    #parses the coordinates for dunky and puts them in a list
    dunkylocations = infile.readline()
    dunkylist = ['' for i in range(int(dunkylocations[0]))]
    grouper = 1
    templist = (dunkylocations.split())
    for i in range(int(dunkylocations[0])):
        dunkylist[i] = [int(templist[grouper]),int(templist[grouper+1])]
        grouper+=2

    runkylist = infile.readline()

    infile.close()
    return pacmap,maprows,mapcols,initdirections,dunkylist,runkylist


class actman:
    def __init__(self, position):
        self.score = 0
        self.startpos = position
        self.position = position
        self.direction = None
        self.movehist = []

    def changePos(self, y, x, newmap):

        #moves actman
        newmap[self.position[0]][self.position[1]] = ' '
        newmap[y][x] = 'A'
        self.position[0] = y
        self.position[1] = x

    def actMove(self, newmap, direction):
        #returns random int modded by length of list of possible moves
        if direction == 'U':
            self.direction = 'U'
            self.changePos(self.position[0]-1,self.position[1],newmap)
        if direction == 'R':
            self.direction = 'R'
            self.changePos(self.position[0],self.position[1]+1,newmap)
        if direction == 'D':
            self.direction = 'D'
            self.changePos(self.position[0]+1,self.position[1],newmap)
        if direction == 'L':
            self.direction = 'L'
            self.changePos(self.position[0],self.position[1]-1,newmap)  
        return newmap
    

    def moneyGet(self, money):
        if self.position in money.nugget_positions:
            self.score += money.nugget_value
            money.nugget_positions.remove(self.position)
        if self.position in money.bar_positions:
            self.score += money.bar_value
            money.bar_positions.remove(self.position)
        if self.position in money.diamond_positions:
            self.score += money.diamond_value
            money.diamond_positions.remove(self.position)

class money:
    def __init__(self, pacmap, maprows, mapcols):
        self.nugget_value = 1
        self.bar_value = 5
        self.diamond_value = 10
        self.nugget_positions = []
        self.bar_positions = []
        self.diamond_positions = []
        self.findMoney(pacmap,maprows, mapcols)


    #finds all positions of the money/score and puts them in respective lists
    def findMoney(self,pacmap, maprows, mapcols):
        for i in range(maprows):
            for j in range(mapcols):
                if pacmap[i][j] == '.':
                    self.nugget_positions.append([i,j])
                if pacmap[i][j] == '$':
                    self.bar_positions.append([i,j])
                if pacmap[i][j] == '*':
                    self.diamond_positions.append([i,j])

    #function that restores money that ghosts have walked over                
    def restoreMoney(self, maprows, mapcols,pacmap):
        entitystr = ['A','P','B','D','R']
        for i in range(maprows):
            for j in range(mapcols):

                ##### restores dots/nuggets that act man has not collected #####
                for k in range(len(self.nugget_positions)):
                    if self.nugget_positions[k][0] == i and self.nugget_positions[k][1] == j and pacmap[i][j] not in entitystr:
                        pacmap[i][j] = '.'

                #### restores dollars/bars that act man has not collected #####
                for l in range(len(self.bar_positions)):
                    if self.bar_positions[l][0] == i and self.bar_positions[l][1] == j and pacmap[i][j] not in entitystr:
                        pacmap[i][j] = '$'

                #### restores stars/diamonds that act man has not collected ######
                for m in range(len(self.diamond_positions)):
                    if self.diamond_positions[m][0] == i and self.diamond_positions[m][1] == j and pacmap[i][j] not in entitystr:

                        pacmap[i][j] = '*'




class punky:
    def __init__(self, position, direction):
        self.position = position
        self.direction = direction


    def changePos(self, y, x, pacmap):

        #keeps track of direction that ghost is moving
        if self.position[0] - y != 0:
            if self.position[0] - y == 1:
                self.direction = 'U'
            elif self.position[0] - y == -1:
                self.direction = 'D'
        if self.position[1] - x != 0:
            if self.position[1] - x == 1:
                self.direction = 'L'
            elif self.position[1] - x == -1:
                self.direction = 'R'
    
        #actually moves the ghost here
        pacmap[self.position[0]][self.position[1]] = ' '
        pacmap[y][x] = 'P'
        self.position[0] = y
        self.position[1] = x

    def punkyMove(self, pacmap, movelist, actpos):
         ################ if punky has 1 possible move ############
        if len(movelist) == 1:
            self.changePos(movelist[0][0], movelist[0][1], pacmap)


        ################# if punky has 2 possible moves ##############
        if len(movelist) == 2:
            movelist = twoPossibleMoves(self.position, self.direction, movelist)

            self.changePos(movelist[0][0], movelist[0][1], pacmap)

        ############## if punky has 3 or more moves #############
        if len(movelist) > 2:
            #calculates the distance between available moves and actman
            for i in range(len(movelist)):
                distance = sqrt(((movelist[i][0] - actpos[0]) ** 2) + ((movelist[i][1] -  actpos[1]) ** 2))
                movelist[i].append(distance)
            
            #sorts list by distance value so first element is always correct move
            movelist.sort(key=lambda x: x[2])
            self.changePos(movelist[0][0], movelist[0][1], pacmap)

                
class bunky:
    def __init__(self, position, direction):
        self.position = position
        self.direction = direction

    def changePos(self, y, x, pacmap):

        #keeps track of direction
        if self.position[0] - y != 0:
            if self.position[0] - y == 1:
                self.direction = 'U'
            elif self.position[0] - y == -1:
                self.direction = 'D'
        if self.position[1] - x != 0:
            if self.position[1] - x == 1:
                self.direction = 'L'
            elif self.position[1] - x == -1:
                self.direction = 'R'

        #actually moves ghost here
        pacmap[self.position[0]][self.position[1]] = ' '
        pacmap[y][x] = 'B'
        self.position[0] = y
        self.position[1] = x

    def bunkyMove(self, pacmap, movelist, actpos, actdir):
         ################ if bunky has 1 possible move ############
        if len(movelist) == 1:
            self.changePos(movelist[0][0], movelist[0][1], pacmap)


        ################# if bunky has 2 possible moves ##############
        if len(movelist) == 2:
            movelist = twoPossibleMoves(self.position, self.direction, movelist)

            self.changePos(movelist[0][0], movelist[0][1], pacmap)
        ############## if bunky has 3 or more possible moves ##########
        if len(movelist) > 2:

            #finds the space 4 cells in from of actman
            if actdir == 'U':
                movey = actpos[0] - 4
                movex = actpos[1]
            if actdir == 'R':
                movey = actpos[0]
                movex = actpos[1] + 4
            if actdir == 'D':
                movey = actpos[0] + 4
                movex = actpos[1]
            if actdir == 'L':
                movey = actpos[0]
                movex = actpos[1] - 4


            #calculates minimum distance betwen possible moves and location 4 cells ahead of actman
            for i in range(len(movelist)):
                distance = sqrt(((movelist[i][0] - movey) ** 2) + ((movelist[i][1] - movex) ** 2))
                movelist[i].append(distance)
                
            #sorts list by distance value so first element is always correct move
            movelist.sort(key=lambda x: x[2])
            self.changePos(movelist[0][0], movelist[0][1], pacmap)
            



class dunky:
    def __init__(self, position, direction):
        self.position = position
        self.direction = direction
        self.pointer = 0

    def changePos(self, y, x, pacmap):

        #keeps track of direction of ghost
        if self.position[0] - y != 0:
            if self.position[0] - y == 1:
                self.direction = 'U'
            elif self.position[0] - y == -1:
                self.direction = 'D'
        if self.position[1] - x != 0:
            if self.position[1] - x == 1:
                self.direction = 'L'
            elif self.position[1] - x == -1:
                self.direction = 'R'

        #changes positon of ghost
        pacmap[self.position[0]][self.position[1]] = ' '
        pacmap[y][x] = 'D'
        self.position[0] = y
        self.position[1] = x

    

    def dunkyMove(self, pacmap, movelist, dunkylist):
        #changes pointer in dunkys patrol list when at specified cell
        if self.position in dunkylist:
            self.pointer += 1
            if self.pointer >= len(dunkylist):
                self.pointer = 0

        ################ if dunky has 1 possible move ############
        if len(movelist) == 1:
            self.changePos(movelist[0][0], movelist[0][1],pacmap)


        ################# if dunky has 2 possible moves ##############
        if len(movelist) == 2:
            movelist = twoPossibleMoves(self.position, self.direction, movelist)

            self.changePos(movelist[0][0], movelist[0][1],pacmap)

        ################ if dunky has more than 2 possible moves ###########
        if len(movelist) > 2:
            targety = dunkylist[self.pointer][0]
            targetx = dunkylist[self.pointer][1]
            
            #calculates distance between possible moves and patrol points
            #loops through to find minimum distance
            for i in range(len(movelist)):
                distance = sqrt(((movelist[i][0] - int(targety)) ** 2) + ((movelist[i][1] - int(targetx)) ** 2))
                movelist[i].append(distance)

            #sorts list by distance value so first element is always correct move
            movelist.sort(key=lambda x: x[2])
            self.changePos(movelist[0][0], movelist[0][1],pacmap)



        
class runky:
    def __init__(self, position, direction):
        self.position = position
        self.pointer = 0
        self.direction = direction


    def changePos(self, y, x, pacmap):
        
        #change direction of ghost
        if self.position[0] - y != 0:
            if self.position[0] - y == 1:
                self.direction = 'U'
            elif self.position[0] - y == -1:
                self.direction = 'D'
        if self.position[1] - x != 0:
            if self.position[1] - x == 1:
                self.direction = 'L'
            elif self.position[1] - x == -1:
                self.direction = 'R'

        #change position of ghost
        pacmap[self.position[0]][self.position[1]] = ' '
        pacmap[y][x] = 'R'
        self.position[0] = y
        self.position[1] = x

    

    def runkyMove(self, pacmap, movelist, runkylist):

        ################ if runky has 1 possible move ############
        if len(movelist) == 1:
            self.changePos(movelist[0][0], movelist[0][1], pacmap)


        ################# if runky has 2 possible moves ##############
        if len(movelist) == 2:
            movelist = twoPossibleMoves(self.position, self.direction, movelist)

            self.changePos(movelist[0][0], movelist[0][1], pacmap)


        ################ if runky has 3 possible moves ###############
        if len(movelist) > 2:
            cont = False
            #loops until runky finds possible move in his list
            while cont == False:
                
                #if current pointer is on UP and he is able to go up
                if runkylist[self.pointer] == 'U' and pacmap[self.position[0] - 1][self.position[1]] != '#':
                    #print('wtf')
                    #print(self.position)
                    self.changePos(self.position[0]-1,self.position[1],pacmap)
                    #print(self.position)
                    self.pointer += 1
                    break

                #if current pointer is right and he is able to go right
                elif runkylist[self.pointer] == 'R' and pacmap[self.position[0]][self.position[1] + 1] != '#':                
                    self.changePos(self.position[0], self.position[1]+1,pacmap)
                    self.pointer += 1
                    break
                
                #if current pointer is down and he is able to go down
                elif runkylist[self.pointer] == 'D' and pacmap[self.position[0] + 1][self.position[1]] != '#':
                    self.changePos(self.position[0] + 1, self.position[1],pacmap)
                    self.pointer += 1
                    break

                #if current pointer is left and he is able to go left
                elif runkylist[self.pointer] == 'L' and pacmap[self.position[0]][self.position[1] - 1] != '#':
                    self.changePos(self.position[0], self.position[1] - 1, pacmap)
                    self.pointer+= 1
                    break
                else:
                    self.pointer += 1
                #loops pointer location in list
                if self.pointer >= len(runkylist):
                    self.pointer = 0


#main game loop
def main():
    pacmap, maprows, mapcols, initdirections, dunkylist, runkylist = parseInput()
    
    init_map = copy.deepcopy(pacmap)

    minscore = 24
    #sequence, newmap, score = actManID_DFS(init_map, maprows,mapcols,initdirections,dunkylist,runkylist)
    sequence, newmap, score = AstarGS(init_map, maprows, mapcols, initdirections, dunkylist, runkylist, minscore)

    #commented out code for BrethFirstSearch
    '''frontier = Queue(maxsize=0)

    
    goal = 16
    frontier.put([])
    while frontier.empty() != True:

        #puts first elemet from frontier into current sequence
        sequence = frontier.get()
        
        print(sequence)
        #calls transition function to get modified map, position of actman, and score
        newmap, position, score = transitionFunction(init_map, sequence, maprows, mapcols, initdirections, dunkylist, runkylist)
        if score >= goal:
            break
        #puts sequence stored into a list into queue
        for i in possibleActMove(position,newmap):
            toput = sequence + i
            frontier.put(toput)'''
    print(sequence)
    print(score)
    pprint.pprint(newmap)

    open(sys.argv[2], 'w').close()
    outfile = open(sys.argv[2], 'a')

    outfile.write(''.join(sequence))
    outfile.write('\n')
    outfile.write(str(score))
    outfile.write('\n')
    outfile.close()
    printPacMap(newmap, maprows, mapcols)


def AstarGS(init_map, maprows, mapcols, initdirections, dunkylist, runkylist, minscore):
    frontier = PriorityQueue(maxsize=0)

    frontier.put((0,[]))
    #keeps track of the number of turns that have passed, used in cost function
    turn = 0

    while frontier.empty() != True:
        sequence = frontier.get()
        print(sequence[1])
        #returns updataed map, position of actman, score, if actman is dead, and if he won
        newmap,position,score,isDead,isWon = transitionFunction(init_map, sequence[1], maprows, mapcols, initdirections, dunkylist, runkylist)

        #checks if actman collected all the points
        if isWon == True:
            return sequence[1], newmap, score

        #checks if actman is dead
        if isDead == True:
            continue

        for i in possibleActMove(position,newmap):
            toput = sequence[1] + i
            tempmap, temppos, hscore, dead, won = transitionFunction(init_map, toput, maprows, mapcols, initdirections, dunkylist, runkylist)
            #adds cost funtion to heuristic function and puts it in the priority queue
            frontier.put(h(toput,hscore+g(turn)))


#Greedy Best First search uses a priority queue which sorts in a way that an action that will get actman closer to the goal will be taken first
def greedyBFS(init_map, maprows, mapcols, initdirections, dunkylist, runkylist, minscore):
    frontier = PriorityQueue(maxsize=0)

    #loads empty list into priority queue
    frontier.put((0, []))

    while frontier.empty() != True:
        sequence = frontier.get()
        print(sequence[1])
        newmap,position,score,isDead = transitionFunction(init_map, sequence[1], maprows, mapcols, initdirections, dunkylist, runkylist)

        #checks if the goal is reached
        if score >= minscore:
            return sequence[1], newmap, score

        #checks if act man died during his last sequence
        if isDead == True:
            continue
        
        #loops through all posible moves that can be taken from act mans current position
        for i in possibleActMove(position,newmap):
            toput = sequence[1] + i
            tempmap, temppos, hscore, dead = transitionFunction(init_map, toput, maprows, mapcols, initdirections, dunkylist, runkylist)
            #calls heuristic function that returns a tuple and puts it in the priority queue
            frontier.put(h(toput, hscore, minscore))



#this is the heuristic function, it negates the score that is passed in because python built in priority queue sorts from least to greatest then puts it in a tuple with the sequence that generated that score
def h(sequence, score):
    sequenceprio =  -score
    return (sequenceprio, sequence)

#this is the cost function, the cost for each step is how may turns it takes to get to that result. The function takes in the number of turns and returns the number of turns plus 1
def g(turn):
    return turn+1



#Itterative deepening depth first search incrementally increases depth and returns the sequence, modified map, and score
def actManID_DFS(initmap, maprows, mapcols, initdirections, dunkylist, runkylist):
    depth = 0
    while True:
        result, newmap, score = actManBounded(initmap, depth, maprows, mapcols, initdirections, dunkylist, runkylist)
        if not isinstance(result, bool):
            return result, newmap, score
        depth += 1
        if result == False:
            break

#Runs DFS while bounded by number passed by ID_DFS functions that returns the sequence, map, and score
def actManBounded(init_map, depth, maprows, mapcols, initdirections, dunkylist, runkylist):
    limit_reach = False

    #creates python built in Lifo Queue
    frontier = LifoQueue(maxsize = 0)
    frontier.put([])
    while frontier.empty() != True:
        sequence = frontier.get()
        print(sequence)

        #calls function that returns map when sequence is preformed on it
        newmap,position,score = transitionFunction(init_map, sequence, maprows, mapcols, initdirections, dunkylist, runkylist)

        #returns when deisred score is reached
        if score >= 20:
            return sequence, newmap, score
        for i in possibleActMove(position,newmap):
            toput = sequence + i

            #checks length of next sequence and if it exceeds depth we dont put it in the frontier
            if len(toput) > depth:
                limit_reach = True
                continue
            frontier.put(toput)
    return limit_reach, newmap, score


#funcion that handles AI movement returns modified map, act man position, and score
def transitionFunction(init_map, sequence, maprows, mapcols, initdirections, dunkylist, runkylist):
    currency = money(init_map, maprows, mapcols)
    tempact = actman(entityFinder('A',maprows,mapcols, init_map))
    dead = False
    gameWon = False
    g1 = punky(entityFinder('P',maprows,mapcols,init_map),initdirections[0])
    g2 = bunky(entityFinder('B',maprows,mapcols,init_map),initdirections[1])
    g3 = dunky(entityFinder('D',maprows,mapcols,init_map),initdirections[2])
    g4 = runky(entityFinder('R',maprows,mapcols,init_map),initdirections[3])
    
    newmap = []
    newmap = copy.deepcopy(init_map)
    if sequence == []:
        return init_map, tempact.position, 0, False, False
    #loops through every move in the current sequence
    for i in sequence:
        newmap = tempact.actMove(newmap,i)

        #if pacman shares the same space as any ghost he dies
        if pacGhostCollision(newmap, tempact, g1, g2, g3, g4):
            dead = True
            break

        #moves punky, bunky, dunky, and runky in that order
        g1.punkyMove(newmap, validMove(g1.position,newmap), tempact.position)
        g2.bunkyMove(newmap, validMove(g2.position,newmap), tempact.position, tempact.direction)
        g3.dunkyMove(newmap, validMove(g3.position,newmap), dunkylist)
        g4.runkyMove(newmap, validMove(g4.position,newmap), runkylist)

        #checks if actman shares same space as ghost again
        if pacGhostCollision(newmap, tempact, g1, g2, g3, g4):
            #print(tempact.position, '\n', g2.position)
            dead = True
            break

        #actman picks up any currency he is on
        tempact.moneyGet(currency)

        if not currency.nugget_positions and not currency.bar_positions and not currency.diamond_positions:
            gameWon = True

        #restores currency that ghosts have walked over
        currency.restoreMoney(maprows,mapcols,newmap)


    return newmap, tempact.position, tempact.score, dead, gameWon




#this function takes in one of the ghosts or actman and decides what move he will take in case of a tie. Returns index of move to be taken
def tieBreaker(gamepiece, movelist):
    for i in range(len(movelist)):
        #up
        if gamepiece.position[0] - 1 == movelist[i][0]:
            return i
    for i in range(len(movelist)):
        #right
        if gamepiece.position[1] + 1 == movelist[i][1]:
            return i
    for i in range(len(movelist)):
        #down
        if gamepiece.position[0] + 1 == movelist[i][0]:
            return i 
    for i in range(len(movelist)):
        if gamepiece.position[1] - 1 == movelist[i][1]:
            return i

#takes in the map and mapsize to print out the map to adhere to assignment specifications
def printPacMap(pacmap, maprows, mapcols):
    outfile = open(sys.argv[2], 'a')
    for i in range(maprows):
        if i > 0:
            outfile.write('\n')
        for j in range(mapcols):
            outfile.write(pacmap[i][j])#,end='')
    outfile.write('\n')


#takes in all gamepieces and checks to see if actman has collided with any of the ghosts. If so returns True
def pacGhostCollision(pacmap, act, punky, bunky, dunky, runky):
    #print(act.position, '\n', bunky.position)
    if act.position == punky.position or act.position == bunky.position or act.position == dunky.position or act.position == runky.position:
        pacmap[act.position[0]][act.position[1]] = 'X'
        return True


#functionality for when the ghosts have two moves in their list of possible moves. makes it to where the will not move opposite their current direction
def twoPossibleMoves(position, direction, movelist):
    if direction == 'U':
        #and object has up in the first index of possible moves
        if position[0] - 1 == movelist[0][0]:
            movelist.pop(1)
        #and object has up in the second index of possible moves
        elif position[0] - 1 == movelist[1][0]:
            movelist.pop(0)
        #object has down as possible move so we remove it 
        elif position[0] + 1 == movelist[0][0]:
            movelist.pop(0)
        else:
            movelist.pop(1)


    if direction == 'D':
        #and object has down in the first index of possible moves
        if position[0] + 1 == movelist[0][0]:
            movelist.pop(1)
        #and object has down in the second index of possible moves
        elif position[0] + 1 == movelist[1][0]:
            movelist.pop(0)
        #object has up direction in moveset so we remove it
        elif position[0] - 1 == movelist[0][0]:
            movelist.pop(0)
        else:
            movelist.pop(1)


    if direction == 'R':
        #and object has right in the first index of possible moves
        if position[1] + 1 == movelist[0][1]:
            movelist.pop(1)
        #and object has right in the second index of possible moves
        elif position[1] + 1 == movelist[1][1]:
            movelist.pop(0)
        #if possible direction is left remove it
        elif position[1] - 1 == movelist[0][1]:
            movelist.pop(0)
        else:
            movelist.pop(1)


    #when objects direction is left        
    if direction == 'L':
        #and object has left in the first index of possible moves
        if position[1] - 1 == movelist[0][1]:
            movelist.pop(1)
        #and object has left in the second index of possible moves
        elif position[1] - 1 == movelist[1][1]:
            movelist.pop(0)
        #if possible direction is right remove it
        elif position[1] + 1 == movelist[0][1]:
            movelist.pop(0)
        else:
            movelist.pop(1)
    return movelist


#returns the location of given gamepiece that is passed to it. Mostly used to find initial location of gamepieces
def entityFinder(gamepiece, maprows, mapcols,pacmap):
    location = ['' for i in range(2)]
    for i in range(maprows):
        for j in range(mapcols):
            if pacmap[i][j] == gamepiece:
                location[0] = i
                location[1] = j
                return location

#creates a list of possible moves for the given gamepiece. Returns that list
def validMove(gamepiece,pacmap):
    movelist = []
    tempplace1 = ['' for i in range(2)]
    tempplace2 = ['' for i in range(2)]
    tempplace3 = ['' for i in range(2)]
    tempplace4 = ['' for i in range(2)]

    #up
    if pacmap[gamepiece[0] - 1][gamepiece[1]] != '#':
        tempplace1[0] = gamepiece[0] - 1
        tempplace1[1] = gamepiece[1]
        movelist.append(tempplace1)
    #right
    if pacmap[gamepiece[0]][gamepiece[1] + 1] != '#':
        tempplace2[0] = gamepiece[0]
        tempplace2[1] = gamepiece[1] + 1
        movelist.append(tempplace2)
    #down
    if pacmap[gamepiece[0] + 1][gamepiece[1]] != '#':
        tempplace3[0] = gamepiece[0] + 1
        tempplace3[1] = gamepiece[1]
        movelist.append(tempplace3)
    #left
    if pacmap[gamepiece[0]][gamepiece[1] - 1] != '#':
        tempplace4[0] = gamepiece[0]
        tempplace4[1] = gamepiece[1] - 1
        movelist.append(tempplace4)
    return movelist

#modified possibleMove function that makes it easier to make act man move. Finds all possible moves and returns them in a list
def possibleActMove(gamepiece, pacmap):#,frontier):
    movelist = []
    tempplace1 = ['' for i in range(2)]
    tempplace2 = ['' for i in range(2)]
    tempplace3 = ['' for i in range(2)]
    tempplace4 = ['' for i in range(2)]
    #up
    if pacmap[gamepiece[0] - 1][gamepiece[1]] != '#':
        tempplace3[0] = gamepiece[0] - 1
        tempplace3[1] = gamepiece[1]
        movelist.append(['U'])
    #right
    if pacmap[gamepiece[0]][gamepiece[1] + 1] != '#':
        tempplace2[0] = gamepiece[0]
        tempplace2[1] = gamepiece[1] + 1
        movelist.append(['R'])
        #print('2')
    #down
    if pacmap[gamepiece[0] + 1][gamepiece[1]] != '#':
        tempplace1[0] = gamepiece[0] + 1
        tempplace1[1] = gamepiece[1]
        movelist.append(['D'])
        #print('1')
    #left
    if pacmap[gamepiece[0]][gamepiece[1] - 1] != '#':
        tempplace4[0] = gamepiece[0]
        tempplace4[1] = gamepiece[1] - 1
        movelist.append(['L'])
        #print('4')
    return movelist
    
#function call of main game loop
main()
