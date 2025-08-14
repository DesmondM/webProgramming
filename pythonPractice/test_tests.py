import pytest
from tests2 import cube

def test_cubing():
    assert cube(3)==27
    assert cube(0) ==0

def test_negCubing():
    assert cube(-2)==-8
    assert cube(-1)==-1