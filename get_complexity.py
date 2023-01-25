#TODO: fix this:
def get_complexity(feat):
    complexity = 0
    for c in feat[1:]:
        if c == '(' or c == '*' or c == '+' or c== '-':
            complexity += 1
    return complexity
