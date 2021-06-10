import matplotlib.pyplot as plt
import constraint as csp
import timeit as t


# Builds the CSP problem for n Queens and returns number of solutions
# or the time to find one solution
def cspQueens(n):
    CSP = csp.Problem()
    Queens = range(1,n+1)               # Queens also repesent columns
    Domains = range(1,n+1)              # Domains are rows
    CSP.addVariables(Queens, Domains)

    start = t.default_timer()

    # Find and all constraints by passing in pairs of queens to be checked
    # for conflicts.
    for Q1 in Queens:
        for Q2 in Queens:
            # This if stament handles column constriants and prevents
            # double checking the same pair in the opposite order
            if Q1 < Q2:
                # Lamda calculus checks for diagonal conflicts and row
                # conflicts then adds conflicting pairs
                CSP.addConstraint(lambda D1, D2, Q1=Q1, Q2=Q2:
                    abs(D1-D2) != abs(Q1-Q2) and D1 != D2, (Q1, Q2))
                      
    CSP.getSolution()
    return t.default_timer() - start

    #solutions = CSP.getSolutions()
    #print(len(solutions))
    #return len(solutions)

if __name__ == "__main__":
    n = 88

    yPlot = []
    for i in range(1,n+1):
        yPlot.append(cspQueens(i))

    numPlot = plt.figure()
    plt.xlabel("Number of Queens (n)")
    plt.ylabel("Time to find first solution (seconds")
    plt.title("Time for n-Queens")
    plt.plot([i for i in range(1,n+1)],yPlot)
    plt.show()
