import pytest

@pytest.mark.parametrize("test_input,output", [
        ('25+1', 26),
        ('5+5', 10),
        ('1*100', 100),
        ('3**2', 9),
        ('1200/12', 100)
    ])
def test_eval(test_input, output):
    assert eval(test_input)==output

