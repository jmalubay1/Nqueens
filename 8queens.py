import random as r
import csv
POP = 50 #Population size, must be even
N = 8 #Number of Queens N, Board size N*N
MUTPCT = 5 #Percent chance to mutate 1-100
GENS = 100


#Creates the initial population of a given size
#Stored as list of lists, each sublist contains
#an 8-character string of ints(0-7) for the queens
#and an int from the fitness function
def newGen(size):
    gen = []
    for i in range(size):
        board = ['',0]
        for j in range(N):
            board[0] += str(r.randint(0,N-1))
        board[1] = fitness(board[0])
        gen.append(board)
    return gen

#Distinct pair attacking:
#Divides the board into rows and diagonals
#Counts the number of queens in each row and each diagonal
#Subtracts sum of conflicts in all rows and diagonals
#with n*(n-1)/2, to exclude duplicate pairs
def fitness(board):
    pairs = (N*(N-1))/2 #Max conflicts
    rows = [0] * N
    diag1 = [0] * 2 * N
    diag2 = [0] * 2 * N
    
    for i in range(N):
        rows[int(board[i])] += 1
        diag1[int(board[i])+i] += 1
        diag2[N-int(board[i])+i] += 1

    for i in range(2*N):
        if i < N:
            pairs -= (rows[i] * (rows[i]-1))/2
        pairs -= (diag1[i] * (diag1[i]-1))/2    
        pairs -= (diag2[i] * (diag2[i]-1))/2
    return int(pairs)

#Populates a list to be randomly chosen from
#Adds the index for each element based on its fitness
#Higher fitness means more chances to be selected
def genProb(gen):
    prob = []
    for i in range(POP):
        for j in range(gen[i][1]):
            prob.append(i)
    return prob

#Takes the generation and probability list and
#Randomly selects two parents and returns their indexes
def getParents(gen,prob):
    parents = [r.choice(prob),r.choice(prob)]
    while parents[0] == parents[1]:
        parents[1] = r.choice(prob)
    return parents

#Creates two children from two parents and a random crossover
#Both children have a random chance to mutate one digit
def makeChild(gen,parents):
    children = ['','']
    cross = r.randint(1,N-1)
    children[0] = gen[parents[0]][0][0:cross] + gen[parents[1]][0][cross:]
    children[1] = gen[parents[1]][0][0:cross] + gen[parents[0]][0][cross:]
    if r.randint(0,100) < MUTPCT:
        mut1 = r.randint(0,N-1)
        mut2 = r.randint(0,N-1)
        temp1 = children[0][:mut1] + str(r.randint(0,N-1)) + children[0][mut1+1:]  
        temp2 = children[1][:mut2] + str(r.randint(0,N-1)) + children[1][mut2+1:]
        children[0] = temp1
        children[1] = temp2
    return children

def buildGen(gen,prob):
    newGen = []
    for i in range(int(POP/2)):
        parents = getParents(gen,prob)
        children = makeChild(gen,parents)
        newGen.append([children[0],fitness(children[0])])
        newGen.append([children[1],fitness(children[1])])
    return newGen
    
        
if __name__ == "__main__":
    gens = []
    gens.append(newGen(POP))
    data = []
    for i in range(1,GENS+1):
        prob = genProb(gens[i-1])
        gens.append(buildGen(gens[i-1],prob))
    for i in range(GENS+1):
        total = 0
        for j in gens[i]:
            total += j[1]
        total = total/POP
        data.append((i,total))
        print(f'Generation: {i} Avg Fitness: {total} ')
    print("4 random first generation states:")
    for i in range(0,4):    
        print(gens[0][r.randint(0,POP-1)])
    print("4 random last generation states:")
    for i in range(0,4):    
        print(gens[GENS][r.randint(0,POP-1)])

    #Remove qoutes to generate a .csv file of average fitness    
    """
    file = str(POP) + "data.csv"
    with open (file, "w", newline="") as csvfile:
        stats = csv.writer(csvfile)
        stats.writerow(("Generation","Avg Fitness"))
        for i in data:
            stats.writerow(i)
    """
