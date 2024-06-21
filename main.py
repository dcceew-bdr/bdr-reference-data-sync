#!/bin/env python3

"""Main entrypoint"""

if __name__ == '__main__':
    import sys
    from src.entrypoint import entrypoint

    sys.exit(entrypoint())
