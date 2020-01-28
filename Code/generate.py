def generate(size, variables):
    """
    Return a random list of instructions.

    :param: size Int
        Number of instructions in the list
    :param: list of String
        List of the names of the variables to use to generate the instructions
    """
    # Number of features available as parameters
    dispo = 0
    
    # Instructions
    l = []

    # Functions that can be used
    FUNCTIONS = {
        1:['sin', 'cos', 'exp'],
        2:['+', '-', '*', '/']
    }
    while len(l) < size:
        if rd.random() > 0.5: # Add a basic variable
            l.append(np.random.choice(variables, 1)[0])
            dispo += 1
        else: # Add a function if there are enough features available to be its parameters
            if rd.random() > 0.5 and dispo >= 1: # Functions with 1 argument
                l.append(np.random.choice(FUNCTIONS[1], 1)[0])
            elif dispo >= 2: # Functions with 2 arguments
                l.append(np.random.choice(FUNCTIONS[2], 1)[0])
                dispo -= 1
    return l