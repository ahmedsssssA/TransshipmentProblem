import numpy as np
import os
# import pandas as pd

def generate_population(LB, UB, pop_size, dim):
    return np.random.uniform(LB, UB,(pop_size, dim))

def modified_bubbleSort(x, y):
    xLen = len(x)
    test = 1
    while test == 1:
        test = 0
        for i in range(xLen-1):
            if x[i] <= x[i+1]:
                tmp = x[i+1]
                tmp2 = np.copy(y[i+1])
                x[i+1] = x[i]
                y[i+1] = np.copy(y[i])
                x[i] = tmp
                y[i] = tmp2
                test = 1
        xLen -= 1
    return x, y

def generateProblem(NS, ND, NT):
    # Generate cost matrix, where the costs belongs to the interval [5, 20]
    # Generate the transportation costs matrix
    CostMatrix = np.random.randint(5, 20, (NS + NT, ND + NT))
    # Generate the fixed costs matrix
    Fixed_costs_matrix = np.random.randint(40, 70, (NS + NT, ND + NT))
    Supplies = np.random.randint(30, 100, NS)
    Demands = np.random.randint(30, 100, ND)
    # Insert dummy and convert transshipment nodes to be represented as supply and demand nodes
    SupSum = sum(Supplies)
    DemSum = sum(Demands)
    if SupSum > DemSum:
        DummyDemand = SupSum - DemSum
        Demands = np.append(Demands, DummyDemand)
        CostMatrix = np.insert(CostMatrix,ND,np.repeat(100000000,NS+NT),axis=1)
        Fixed_costs_matrix = np.insert(Fixed_costs_matrix,ND,np.repeat(100000000,NS+NT),axis=1)
    elif SupSum < DemSum:
        DummySupply = DemSum - SupSum
        Supplies = np.append(Supplies, DummySupply)
        CostMatrix = np.insert(CostMatrix,NS,np.repeat(100000000,ND+NT),axis=0)
        Fixed_costs_matrix = np.insert(Fixed_costs_matrix,NS,np.repeat(100000000,ND+NT),axis=0)

    Transshipment = np.ones(NT) * sum(Supplies)
    solution = np.zeros((len(Supplies)+ NT, len(Demands) + NT))
    Supplies = np.append(Supplies, Transshipment)
    Demands = np.append(Demands, Transshipment)
    indexOrderedProduct = np.array([(i, j) for i in range(solution.shape[0]) for j in range(solution.shape[1])])
    return solution, indexOrderedProduct, Supplies, Demands, CostMatrix, Fixed_costs_matrix

def generateSolution(rndStructure, indexOrderedProduct, solution, CostMatrix, Fixed_costs_matrix, Supplies, Demands):
    SolStruct = np.copy(indexOrderedProduct)
    solution = np.copy(solution)
    Supplies = np.copy(Supplies)
    Demands = np.copy(Demands)
    modified_bubbleSort(rndStructure, SolStruct)
    while len(SolStruct) > 0:
        # The row represents the supply and the columns represens the demand
        row = SolStruct[0][0]
        column = SolStruct[0][1]
        if Supplies[row] == Demands[column]:
            solution[row, column] = Supplies[row]
            Demands[column] = 0
            Supplies[row] = 0
            SolStruct = np.array([(i[0], i[1]) for i in SolStruct if i[0] != row])
            SolStruct = np.array([(i[0], i[1]) for i in SolStruct if i[1] != column])
        elif Supplies[row] <= Demands[column]:
            solution[row, column] = Supplies[row]
            Demands[column] = Demands[column] - Supplies[row]
            Supplies[row] = 0
            SolStruct = np.array([(i[0], i[1]) for i in SolStruct if i[0] != row])
        else:
            solution[row, column] = Demands[column]
            Supplies[row] = Supplies[row] - Demands[column]
            Demands[column] = 0
            SolStruct = np.array([(i[0], i[1]) for i in SolStruct if i[1] != column])
    solFixed = solution.copy()
    solFixed[solFixed > 1] = 1
    objfixedvalues = sum([j for j in np.nditer(solution * Fixed_costs_matrix) if j < 100000000])
    objectiveValue = sum([j for j in np.nditer(solution * CostMatrix) if j < 100000000])
    objectiveValue += objfixedvalues 
    # objectiveValue = sum(sum(solution @ CostMatrix.T)) - max(sum(solution @ CostMatrix.T))
    return solution, objectiveValue

# Relocate penguin position
def Relocate(P, P_other, Itr, MaxItr, LB, UB, R, thershold = 0.4):
    # Compute the temperature profile around the huddle
    R = np.random.uniform(0, R)
    Temp = temp(Itr, MaxItr, R)
    # The polygon grid accuracy
    P_grid = abs(P - P_other)
    # Calculate avoid collision parameter
    A = 2 * (Temp + P_grid) * np.random.uniform() - Temp
    # The social forces of emperor penguins
    S = np.sqrt(np.random.uniform(5, 50) * np.exp(-Itr/np.random.uniform(5, 50)) - np.exp(-Itr))
    # Relocate penguin
    D = abs(S * P - np.random.uniform() * P_other)
    modified = fix_bounds(P - A * D, LB, UB)
    newposition = np.array([])
    for i in range(len(modified)):
        if np.random.uniform() > thershold:
            newposition = np.append(newposition,P[i])
        else:
            newposition = np.append(newposition, modified[i])
    return newposition

def PCA(LB, UB, dim, pop_size, MaxItr, R, indexOrderedProduct, solution, CostMatrix, Fixed_costs_matrix, Supplies, Demands):

    # Initialize the emperor penguins population
    population = generate_population(LB, UB, pop_size, dim)
    # Calculate the fitness value of each search agent
    fitness = [generateSolution(x, indexOrderedProduct, solution, CostMatrix, Fixed_costs_matrix, Supplies, Demands)[1] for x in population]
    Gbest = population[fitness.index(min(fitness)),:]
    Gbest_sol = generateSolution(Gbest, indexOrderedProduct, solution, CostMatrix, Fixed_costs_matrix, Supplies, Demands)
    best = min(fitness)
    P_vector_1 = np.array([])
    P_vector_2 = np.arange(MaxItr)
    for i in range(MaxItr):
        for j in range(pop_size):
            # Update position
            Current_Position = population[j]
            Current_Position = Relocate(Gbest, Current_Position, i, MaxItr, LB, UB, R)
            Solution = generateSolution(Current_Position, indexOrderedProduct, solution, \
                CostMatrix, Fixed_costs_matrix, Supplies, Demands)
            fitness[j] = Solution[1]

            # Update the position of the optimal solution
            if fitness[j] <= best:
                best = fitness[j]
                Gbest = Current_Position
                Gbest_sol = Solution[0]
        P_vector_1 = np.append(P_vector_1,best)

    return best, Gbest, Gbest_sol, P_vector_2, P_vector_1

# Fix bounds function
def fix_bounds(x, LB, UB):
    x_lenght = len(x)
    for i in range(x_lenght):
        if x[i] > UB or x[i] < LB:
            x[i] = np.random.uniform(LB, UB)
    return x

# The temperature profile around the huddle
def temp(Itr, MaxItr, R):
    if R >= 1:
        T = 0
    else:
        T = 1
    return T - MaxItr/(Itr - MaxItr)


def generate_problemData(NS, ND, NT):
    # Generate cost matrix, where the costs belongs to the interval [5, 20]
    # Generate the transportation costs matrix
    CostMatrix = np.random.randint(5, 20, (NS + NT, ND + NT))
    # Generate the fixed costs matrix
    Fixed_costs_matrix = np.random.randint(40, 70, (NS + NT, ND + NT))
    Supplies = np.random.randint(30, 100, NS)
    Demands = np.random.randint(30, 100, ND)
    return(Supplies, Demands, Fixed_costs_matrix, CostMatrix)

def ConfigureProblem(Supplies, Demands, CostMatrix, Fixed_costs_matrix):
    NS = len(Supplies)
    ND = len(Demands)
    NT = CostMatrix.shape[0] - NS
    # Insert dummy and convert transshipment nodes to be represented as supply and demand nodes
    SupSum = sum(Supplies)
    DemSum = sum(Demands)
    if SupSum > DemSum:
        DummyDemand = SupSum - DemSum
        Demands = np.append(Demands, DummyDemand)
        CostMatrix = np.insert(CostMatrix,ND,np.repeat(100000000,NS+NT),axis=1)
        Fixed_costs_matrix = np.insert(Fixed_costs_matrix,ND,np.repeat(100000000,NS+NT),axis=1)
    elif SupSum < DemSum:
        DummySupply = DemSum - SupSum
        Supplies = np.append(Supplies, DummySupply)
        CostMatrix = np.insert(CostMatrix,NS,np.repeat(100000000,ND+NT),axis=0)
        Fixed_costs_matrix = np.insert(Fixed_costs_matrix,NS,np.repeat(100000000,ND+NT),axis=0)

    Transshipment = np.ones(NT) * sum(Supplies)
    solution = np.zeros((len(Supplies)+ NT, len(Demands) + NT))
    Supplies = np.append(Supplies, Transshipment)
    Demands = np.append(Demands, Transshipment)
    indexOrderedProduct = np.array([(i, j) for i in range(solution.shape[0]) for j in range(solution.shape[1])])
    return solution, indexOrderedProduct, Supplies, Demands, CostMatrix, Fixed_costs_matrix

def generateProblems(NS, ND, NT, numberOfProblems):
    for i in range(numberOfProblems):
        problem = generate_problemData(NS, ND, NT)
        with open('problems' + str(i) + '.csv', 'w') as fl:
            supplies = problem[0]
            demands = problem[1]
            CostMatrix = problem[2]
            fixedCosts = problem[3]
            fl.write('Supplies\n')
            for j in supplies:
                fl.write(str(j)+ '\n')
            fl.write('Demands\n')
            for j in demands:
                fl.write(str(j)+ '\n')
            fl.write('CostMatrix\n')
            for j in range(CostMatrix.shape[0]):
                text = ",".join([str(CostMatrix[j,:][k]) for k in range(CostMatrix.shape[1])])
                fl.write(text + '\n')
            fl.write('fixedCosts\n')
            for j in range(fixedCosts.shape[0]):
                text = ",".join([str(fixedCosts[j,:][k]) for k in range(fixedCosts.shape[1])])
                fl.write(text + '\n')

def get_problem(problem, ProblemDir):
    with open(os.getcwd() + f'/{ProblemDir}/{problem}.csv', 'r') as fl:
        problem = fl.readlines()
        Supplies = []
        Demands = []
        cnt = 0
        for i in problem:
            if i == 'Demands\n':
                demandline = cnt
            elif i == 'CostMatrix\n':
                costmatrixline = cnt
            elif i == 'fixedCosts\n':
                fixedcostsline = cnt
            cnt +=1
        for j in range(1, demandline):
            Supplies.append(float(problem[j].strip()))
        for j in range(demandline + 1, costmatrixline):
            Demands.append(float(problem[j].strip()))
        try:
            del CostMatrix
            for j in range(costmatrixline + 1, fixedcostsline):
                row = np.array([[int(float(x.strip())) for x in problem[j].split(',')]])
                try:
                    CostMatrix = np.append(CostMatrix, row, axis=0)
                except NameError:
                    CostMatrix = row
        except NameError:
            for j in range(costmatrixline + 1, fixedcostsline):
                row = np.array([[int(float(x.strip())) for x in problem[j].split(',')]])
                try:
                    CostMatrix = np.append(CostMatrix, row, axis=0)
                except NameError:
                    CostMatrix = row
        try:
            del fixedCosts
            for j in range(fixedcostsline + 1, len(problem)):
                row = np.array([[int(float(x.strip())) for x in problem[j].split(',')]])
                try:
                    fixedCosts = np.append(fixedCosts, row, axis=0)
                except NameError:
                    fixedCosts = row
        except NameError:
            for j in range(fixedcostsline + 1, len(problem)):
                row = np.array([[int(float(x.strip())) for x in problem[j].split(',')]])
                try:
                    fixedCosts = np.append(fixedCosts, row, axis=0)
                except NameError:
                    fixedCosts = row
    return(Supplies, Demands, CostMatrix, fixedCosts)