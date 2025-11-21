from pathbench.utils.registry import Registry


def test_registry():
r = Registry()
@r.register("a")
def f(): return 1
assert r.get("a")() == 1