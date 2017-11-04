import itertools


def peek(iterable):
    it = iter(iterable)
    try:
        first = next(it)
    except StopIteration:
        return None
    return itertools.chain([first], it)


def first(iterable):
    it = iter(iterable)
    try:
        return next(it)
    except StopIteration:
        return None

def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)