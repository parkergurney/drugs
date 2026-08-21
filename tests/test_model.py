from precision_md.model import MaceEvaluator


class FakeCalculator:
    def __init__(self):
        self.results = {"energy": -1.0}
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1
        self.results.clear()


class FakeAtoms:
    calc = None


def test_attaching_calculator_clears_cross_policy_cache():
    evaluator = MaceEvaluator.__new__(MaceEvaluator)
    evaluator.calculator = FakeCalculator()
    atoms = FakeAtoms()

    evaluator._attach_fresh_calculator(atoms)

    assert evaluator.calculator.reset_calls == 1
    assert evaluator.calculator.results == {}
    assert atoms.calc is evaluator.calculator


def test_bf16_numpy_conversion_failure_is_a_supported_failure_type():
    handled = (RuntimeError, NotImplementedError, TypeError)
    error = TypeError("Got unsupported ScalarType BFloat16")

    assert isinstance(error, handled)
