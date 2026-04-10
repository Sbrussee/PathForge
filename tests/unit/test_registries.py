from pathbench.utils.registry import Registry


def test_registry():
    registry = Registry()

    @registry.register("a")
    def factory():
        return 1

    assert registry.get("a")() == 1
