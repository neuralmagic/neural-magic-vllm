#!/usr/bin/env python3

import coverage
import pathlib
import pytest


def find_all_tests(path='tests/'):
    return pathlib.Path(path).rglob('*test_*')


def body():
    all_test_files = find_all_tests()
    for file in all_test_files:
        print(f"{file}")


if __name__ == "__main__":
    body()
