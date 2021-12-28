"""6.009 Lab 8B: carlae Interpreter Part II"""

import sys
import doctest
# NO ADDITIONAL IMPORTS!
# IN ADDITION, DO NOT USE sys OTHER THAN FOR THE THINGS DESCRIBED IN THE LAB WRITEUP


class EvaluationError(Exception):
    """Exception to be raised if there is an error during evaluation."""
    pass


def tryapp(result, text):
    ''' A helper function that takes in the result list and a string and appends it if not empty. 
    Returns an empty string to reset temp. '''
    if len(text)!=0:
        result.append(text)
    return ''

def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a carlae
                      expression
    """
    result = []
    for line in source.splitlines(): # Go through source line by line
        temp = ''
        for char in line: # Go through each char
            if char in {'(',')'}: # Recognized operand
                temp = tryapp(result, temp)
                result.append(char)
            elif char==';': # Break
                temp = tryapp(result, temp)
                break
            elif char==' ': # Space, append?
                temp = tryapp(result, temp)
            else: # It's a letter
                temp += char
        tryapp(result, temp)
    return result

def number(first):
    ''' A helper function that takes in a string and attempts to covert it into an int or float.'''
    result = True
    if '.' in first: # It's a float?
        for char in first:
            if char not in {'.','-'} and not char.isdigit():
                result = False
                break
        if result:
            return float(first)
    elif first.isdigit() or ('-' in first and first!='-'): # It's an int or neg
        for char in first:
            if char!='-' and not char.isdigit():
                result = False
                break
        if result:
            return int(first)
    return first

def preparse(tokens):
    ''' A helper function that recursively parses the tokens and returns the result and tokens left.'''
    result = []
    openpar = False # To keep track of parentheses
    while len(tokens)>0:
        first = tokens[0]
        tokens = tokens[1:]

        if first=='(': # Recursion!
            if openpar: # Not first run
                text, tokens = preparse([first]+tokens)
                result.append(text)
            else:
                openpar = True
        elif first==')':
            if openpar:
                if len(tokens)!=0: # Not last run
                    return result, tokens
                openpar = False
            else:
                raise SyntaxError
        
        else: # Number or string
            if not openpar and tokens[0:1]!='(' and tokens!=[]:
                raise SyntaxError
            text = number(first)
            result.append(text)

    if openpar:
        raise SyntaxError
    return result, []
        
def parse(tokens):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    """
    result, left = preparse(tokens) # Get the result and leftovers

    if len(result)==len(tokens): # If no parentheses
        if len(result)==1:
            result = result[0]
        else:
            raise SyntaxError

    if len(left)==0: # Worked
        return result
    raise SyntaxError

def mul(args):
    ''' Multiplies the first entry by all others. '''
    for each in args[1:]:
        args[0] *= each
    return args[0]

def div(args):
    ''' Divides the first entry by all others. '''
    for each in args[1:]:
        args[0] /= each
    return args[0]

def define(tree, envir):
    ''' A helper function that defines variables within the given environment. '''
    if type(tree[0])==list: # Change to function def
        params = tree[0][1:]
        tree[0] = tree[0][0]
        tree[1] = ['lambda', params, tree[1]]
    envir[tree[0]] = tree[1]
    return envir[tree[0]]

def eq(args):
    ''' A helper function that checks to see if everything's equal. '''
    for each in args:
        if each!=args[0]:
            return False
    return True
    
def greater(args):
    ''' A helper function that checks that arguments are in decreasing order. '''
    for each in args[1:]:
        if each>=args[0]:
            return False
    return True

def greatereq(args):
    ''' A helper function that checks that arguments are in decreasing order or the same. '''
    for each in args[1:]:
        if each>args[0]:
            return False
    return True

def less(args):
    ''' A helper function that checks that arguments are in increasing order. '''
    for each in args[1:]:
        if each<=args[0]:
            return False
    return True

def lesseq(args):
    ''' A helper function that checks that arguments are in increasing order or the same. '''
    for each in args[1:]:
        if each<args[0]:
            return False
    return True

def andd(args, envir):
    ''' A helper function that checks that args are true. '''
    for each in args:
        if not evaluate(each, envir):
            return False
    return True

def orr(args, envir):
    ''' A helper function that checks that at least one arg is true. '''
    for each in args:
        if evaluate(each, envir): 
            return True
    return False

def nott(args):
    ''' A helper function that returns the opposite of the given logic. '''
    return not args[0]

def iff(tree, envir):
    ''' A helper function that deals with if statements. '''
    if evaluate(tree[0], envir):
        return evaluate(tree[1], envir)
    return evaluate(tree[2], envir)

class env():
    ''' A class for environments.'''
    def __init__(self, par=None):
        if not par: # Use default parent
            self.parent = carlae_builtins
        else:
            self.parent = par

        self.vars = {}

    def __setitem__(self, name, val):
        ''' Creates a new attribute with the given exp as value.'''
        self.vars[name] = evaluate(val, self)

    def __getitem__(self, name):
        ''' Looks for the variable and returns value if found. '''
        if name in self.vars: # Found
            return self.vars[name]
        elif name in self.parent:
            return self.parent[name]
        else:
            raise EvaluationError

    def __contains__(self, name):
        ''' Searches for the given variable. '''
        if name in self.vars: # Found
            return True
        elif name in self.parent:
            return True
        else:
            return False

    def sett(self, name, val):
        ''' Looks for the name and then sets it to the new val.'''
        if name in self.vars: # Found
            self.vars[name]=val
        elif self.parent!=carlae_builtins:
            self.parent.sett(name, val)
        else:
            raise EvaluationError

class function():
    ''' A class to hold function objects. '''
    def __init__(self, temp, envir):
        self.params = temp[0]
        self.exp = temp[1]
        self.envir = envir

    def __call__(self, params):
        envir = env(self.envir)
        if len(self.params)!=len(params): # Wrong num of arguments
            raise EvaluationError
        for i in range(len(params)): # Define all the variables in the local environment
            evaluate(['define', self.params[i], params[i]], envir)
        return evaluate(self.exp, envir)

class Pair():
    def __init__(self,car,cdr):
        self.car = car
        self.cdr = cdr

    def copy(self):
        ''' Makes a deep copy of itself. '''
        if type(self.cdr)==Pair:
            return Pair(self.car, self.cdr.copy())
        return Pair(self.car, self.cdr)

    def __iter__(self):
        ''' Iterates through the Carlae list. '''
        yield self.car
        if type(self.cdr)==Pair:
            yield from self.cdr

def cons(tree):
    ''' A function that creates new Pair instances. '''
    return Pair(tree[0], tree[1])

def car(arg):
    ''' A function that returns the car variable of the Pair instance. '''
    if type(arg[0])==Pair: # Pair instance
        return arg[0].car
    raise EvaluationError

def cdr(arg):
    ''' A function that returns the cdr variable of the Pair instance. '''
    if type(arg[0])==Pair: # Pair instance
        return arg[0].cdr
    raise EvaluationError

def listt(tree):
    ''' A function that creates lists. '''
    if tree==[]:
        return []
    else:
        first = tree[0]
        if first==[]:
            first = 'nil'
        second = listt(tree[1:])
        second = [] if second==[] else second
    return cons([first, second])

def length(tree):
    ''' A function that returns the length of the list. '''
    if tree[0]==[] or tree[0]=='nil':
        return 0
    elif type(tree[0])!=Pair:
        raise EvaluationError
    if type(tree[0].cdr)==Pair: # Not end yet
        return 1 + length([tree[0].cdr])
    elif tree[0].cdr==[] or tree[0].cdr=='nil': # End of list
        return 1
    raise EvaluationError

def index(tree):
    ''' A function that indexes into the list. '''
    var, i = tree
    if type(var)!=Pair:
        raise EvaluationError
    if i==0:
        return var
    if type(var.cdr)!=Pair:
        raise EvaluationError
    return index([var.cdr,i-1])

def icar(tree):
    ''' A function that returns the car of the indexed list element.'''
    return index(tree).car

def concat(tree):
    ''' A function that concatenates lists. '''
    if len(tree)==0 or tree=='nil':
        return []
    result = tree[0]
    if result==[] or result=='nil':
        return concat(tree[1:])
    if type(result)!=Pair:
        raise EvaluationError
    new = result.copy() # Makes a copy
    l = length([new])
    index([new,l-1]).cdr = concat(tree[1:])
    return new

def mapp(tree):
    ''' A function that maps the function to the given arguments. '''
    func, args = tree
    if not callable(func) or (type(args)!=Pair and args!=[]):
        raise EvaluationError
    result = listt([]) # Start with an empty list
    for each in args:
        result = concat([result, listt([func([each])])])
    return result

def filter(tree):
    ''' A function that filters values that makes the cond not true. '''
    func, args = tree
    if not callable(func) or (type(args)!=Pair and args!=[]):
        raise EvaluationError
    result = listt([]) # Start with an empty list
    for each in args:
        temp = func([each])
        if temp:
            result = concat([result, listt([each])])
    return result

def reduce(tree):
    ''' A function that applies the given function to the initial value and list.'''
    func, args, init = tree
    if not callable(func) or (type(args)!=Pair and args!=[]):
        raise EvaluationError
    for each in args:
        init = func([init,each])
    return init

def begin(tree):
    ''' A function that returns the last value. '''
    return tree[-1]

def let(tree, envir):
    ''' A function that assigns variables and evaluates the body.'''
    new = env(envir)
    variables, body = tree
    for each in variables:
        define(each, new)
    return evaluate(body, new)

def sett(tree, envir):
    ''' A function that assigns a variable the evaluated expression. '''
    var, expr = tree
    envir.sett(var, evaluate(expr, envir))
    return envir[var]

carlae_builtins = {
    '+': sum,
    '-': lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    '*': mul,
    '/': div,
    '#t':True,
    '#f':False,
    'nil':[],
    '=?':eq,
    '>':greater,
    '>=':greatereq,
    '<':less,
    '<=':lesseq,
    'not':nott,
    'cons':cons,
    'car':car,
    'cdr':cdr,
    'list':listt,
    'length':length,
    'elt-at-index':icar,
    'concat':concat,
    'map':mapp,
    'filter':filter,
    'reduce':reduce,
    'begin':begin,
}

special = {
    'lambda': function,
    'define': define,
    'if': iff,
    'and':andd,
    'or':orr,
    'let':let,
    'set!':sett,
}

def evaluate_file(name, envir=False):
    ''' A function that opens files. '''
    if not envir:
        envir = env()
    with open(name, encoding="utf-8") as f:
        text = f.read()
    tokens = tokenize(text)
    tree = parse(tokens)
    return evaluate(tree, envir)

def evaluate(tree, envir=False):
    """
    Evaluate the given syntax tree according to the rules of the carlae
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """
    if not envir:
        envir = env()
    
    if type(tree)==list:
        if len(tree)==0:
            raise SyntaxError
        if type(tree[0])==str and tree[0] in special: # For special forms
            return special[tree[0]](tree[1:], envir)
        tree = [evaluate(x, envir) for x in tree]
        if callable(tree[0]):
            return tree[0](tree[1:])
    elif tree in envir:
        return envir[tree]
    elif type(tree) in {int,float,function,bool,Pair}:
        return tree
    raise EvaluationError
                
def result_and_env(tree, envir=None):
    ''' Automatically checks the result and environment. '''
    if not envir:
        envir = env()
    return (evaluate(tree,envir), envir)



if __name__ == '__main__':
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)

    # uncommenting the following line will run doctests from above
    # doctest.testmod()
    test = ['+', 3, ['-', 7, 5]]
    print(evaluate(test))
    print(evaluate('+'))
    print(evaluate(3.14))

    test = ['begin', [
    "define",
    [
      "allprimes",
      "n"
    ],
    [
      "allprimes-iter",
      "n",
      [
        "list"
      ]
    ]
  ],
  [
    "define",
    [
      "prime-iter",
      "n",
      "m"
    ],
    [
      "if",
      [
        ">=",
        "n",
        "m"
      ],
      "#t",
      [
        "if",
        [
          "divides?",
          "m",
          "n"
        ],
        "#f",
        [
          "prime-iter",
          "n",
          [
            "+",
            "m",
            1
          ]
        ]
      ]
    ]
  ],
  [
    "define",
    [
      "prime?",
      "n"
    ],
    [
      "if",
      [
        "<",
        "n",
        2
      ],
      "#f",
      [
        "prime-iter",
        "n",
        2
      ]
    ]
  ],
  [
    "define",
    [
      "prime-iter",
      "n",
      "m"
    ],
    [
      "if",
      [
        ">=",
        "m",
        "n"
      ],
      "#t",
      [
        "if",
        [
          "divides?",
          "m",
          "n"
        ],
        "#f",
        [
          "prime-iter",
          "n",
          [
            "+",
            "m",
            1
          ]
        ]
      ]
    ]
  ],
  [
    "define",
    [
      "divides?",
      "x",
      "y"
    ],
    [
      "if",
      [
        "=?",
        "y",
        0
      ],
      "#t",
      [
        "if",
        [
          "<",
          "y",
          0
        ],
        "#f",
        [
          "divides?",
          "x",
          [
            "-",
            "y",
            "x"
          ]
        ]
      ]
    ]
  ],
  [
    "define",
    [
      "allprimes-iter",
      "n",
      "sofar"
    ],
    [
      "if",
      [
        "<",
        "n",
        0
      ],
      "sofar",
      [
        "let",
        [
          [
            "isprime",
            [
              "prime?",
              "n"
            ]
          ]
        ],
        [
          "let",
          [
            [
              "newlist",
              [
                "if",
                "isprime",
                [
                  "concat",
                  "sofar",
                  [
                    "list",
                    "n"
                  ]
                ],
                "sofar"
              ]
            ]
          ],
          [
            "allprimes-iter",
            [
              "-",
              "n",
              1
            ],
            "newlist"
          ]
        ]
      ]
    ]
  ],
  [
    "allprimes",
    30
  ]]
    result = evaluate(test)
    print(list(result))

    envir = env()
    # for name in sys.argv: # Evaluate the given files
    #     evaluate_file(name, envir)
    while True:
        text = input('Input:')
        if text=='QUIT':
            break
        tokens = tokenize(text)
        try:
            tree = parse(tokens)
            try:
                result = evaluate(tree, envir)
                print('Result:',result)
            except EvaluationError:
                print('Evaluation Error')
        except SyntaxError:
            print('SyntaxError')

    pass
