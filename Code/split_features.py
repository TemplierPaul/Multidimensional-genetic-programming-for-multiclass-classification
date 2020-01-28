def split_features(instructions, verbose=False):
    """
    Splits an instructions list into sub-lists of 1 feature each.

    :param: instructions
        list of instructions to split
    :param: verbose
        Default False
        If True, prints the state of the algorithm at each step
    :return: list of sub-lists, each with the instructions for 1 feature
    """
    l = np.flip(instructions)
    s = []  # list of features
    buffer = []  # Intermediate list of nodes in the feature
    depth = 1  # Number of leaves needed to complete the tree
    for k in range(len(l)):
        i = l[k]
        buffer.append(i)

        if i in Var.FUNCTIONS[1]:  # Functions that take 1 argument (eg sin, cos, exp) have as many parents as children
            depth += 0
        elif i in Var.FUNCTIONS[2]:  # Functions that take 2 arguments (eg +, -, *, /) have 2 children but 1 parent,
            # hence they increment the number of leaves needed in the tree
            depth += 1
        else:  # Other values are variables, which decrement the number of leaves needed
            depth -= 1

        # If the sub-tree in the buffer needs no more leaves, it is complete.
        # It is then saved in the returned list and the buffer is cleared.
        # The new sub-tree starts with at least 1 node, hence depth=1
        if depth <= 0:
            s.append(np.flip(buffer).tolist())
            buffer = []
            depth = 1
        if verbose:
            print(k, i, ' ' * (8 - len(str(i))) + '| b=', buffer, '| s=', s)
    s = np.flip(s, axis=0).tolist()
    return s