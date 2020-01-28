class Var(object):
    """
    Variable used to translate a list of instructions into a feature.
    Var objects support standard operations (eg +, -, *, /, cos, sin, exp) and apply them on the dataset.

    name: String
        Default: None
        Name of the column in the initial dataset used by this variable
    funct: String
        Default: None
        If this variable is a transformation of 1 (or more) variables, self.funct is the name of the function it implements
    params: list of Var
        Default: None
        List of variables used by the function it implements
    data: panda.Series
        Default: None
        Column of data transformed from the initial dataset with the feature implemented
    seize: Integer in {0, 1, 2}
        Default: 0
        Number of parameters needed by the function
    depth: Integer
        Default: 0 (for a column in the initial dataset with no transformation)
        Depth of the tree from this node
    """

    def __init__(self, name=None, data=None):
        self.name = name
        self.funct = None
        self.params = None
        self.data = data
        self.size = 0
        self.depth = 0

    def __str__(self):
        """
        Description of the object: name of the column or operation it implements.
        Recursive function on the params list if not empty.

        :return: String
        """
        if self.funct is None:
            return self.name
        else:
            if self.size == 1:
                return self.funct + '(' + str(self.params[0]) + ')'
            elif self.size == 2:
                s = ''
                if self.params[0].size == 2:
                    s += '(' + str(self.params[0]) + ')'
                else:
                    s += str(self.params[0])
                s += ' ' + self.funct + ' '
                if self.params[1].size == 2:
                    s += '(' + str(self.params[1]) + ')'
                else:
                    s += str(self.params[1])
                return s

        return "Error"

    def __repr__(self):
        """
        Detailed description for debug.

        :return: String
        """
        r = '> '
        r += 'depth= ' + str(self.depth)
        r += ' |  ' + str(self)
        return r

    def __add__(self, other):
        """
        Addition

        :param: other
            Other Var to add
        :return: Var
            New Var implementing this operation
        """
        v = Var()
        v.funct = '+'
        v.params = [self, other]
        v.size = 2
        v.depth = max(self.depth, other.depth) + 1
        return v

    def __sub__(self, other):
        """
        Substraction

        :param: other
            Other Var to add
        :return: Var
            New Var implementing this operation
        """
        v = Var()
        v.funct = '-'
        v.params = [self, other]
        v.size = 2
        v.depth = max(self.depth, other.depth) + 1
        return v

    def __mul__(self, other):
        """
        Multiplication

        :param: other
            Other Var to add
        :return: Var
            New Var implementing this operation
        """
        v = Var()
        v.funct = '*'
        v.params = [self, other]
        v.size = 2
        v.depth = max(self.depth, other.depth) + 1
        return v

    def __truediv__(self, other):
        """
        Division

        :param: other
            Other Var to add
        :return: Var
            New Var implementing this operation
        """
        v = Var()
        v.funct = '/'
        v.params = [self, other]
        v.size = 2
        v.depth = max(self.depth, other.depth) + 1
        return v

    def sin(self):
        """
        Sine

        :return: Var
            New Var implementing this operation
        """
        v = Var()
        v.funct = 'sin'
        v.params = [self]
        v.size = 1
        v.depth = self.depth + 1
        return v

    def cos(self):
        """
        Cosine

        :return: Var
            New Var implementing this operation
        """
        v = Var()
        v.funct = 'cos'
        v.params = [self]
        v.size = 1
        v.depth = self.depth + 1
        return v

    def exp(self):
        """
        Exponential

        :return: Var
            New Var implementing this operation
        """
        v = Var()
        v.funct = 'exp'
        v.params = [self]
        v.size = 1
        v.depth = self.depth + 1
        return v
    
    def log(self):
        """
        Logarithm

        :return: Var
            New Var implementing this operation
        """
        v = Var()
        v.funct = 'log'
        v.params = [self]
        v.size = 1
        v.depth = self.depth + 1
        return v

    def compute(self, data=None):
        """
        Function computing the new feature from the initial dataset.
        Reccursive function: calls itself on Var objects in params

        :param: data
            Default: None
            pandas.DataFrame with the initial dataset
            If None, uses self.data
        :return: np.array
            transformed data
        """
        if self.funct is None:
            if data is not None:
                self.data = data[self.name]
            assert self.data is not None
            d = self.data
        else:
            if self.funct == '+':
                d = self.params[0].compute(data=data) + self.params[1].compute(data=data)
            elif self.funct == '-':
                d = self.params[0].compute(data=data) - self.params[1].compute(data=data)
            elif self.funct == '*':
                d = self.params[0].compute(data=data) * self.params[1].compute(data=data)
            elif self.funct == '/':
                d = self.params[0].compute(data=data) / self.params[1].compute(data=data)
            elif self.funct == 'sin':
                d = np.sin(self.params[0].compute(data=data))
            elif self.funct == 'cos':
                d = np.cos(self.params[0].compute(data=data))
            elif self.funct == 'log':
                d = np.log(self.params[0].compute(data=data))
            elif self.funct == 'exp':
                d = np.exp(self.params[0].compute(data=data))
        return d

    # List of functions implementable with this class
    # FUNCTIONS[i][funct] with i in {1, 2} and funct a String will yield the method used to implement funct with i parameters
    FUNCTIONS = {
        1: {
            'sin': sin,
            'cos': cos
        },
        2: {
            '+': __add__,
            '-': __sub__,
            '*': __mul__
        }
    }