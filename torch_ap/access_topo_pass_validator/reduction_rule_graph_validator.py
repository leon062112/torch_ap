import torch
import torch.fx as fx
from typing import Any
from torch_ap.spider import up_spider, down_spider


class AccessTopoReductionRuleGraphValidator:
    """
    AccessTopoReductionRuleGraphValidator := void <- $pattern_func <- $replacement_func <- ()
    """

    def __init__(self):
        # # inline: $tracer fx.Tracer
        # Explicitly autowrap spider functions to capture them as nodes in the graph.
        self.tracer = fx.Tracer(autowrap_functions=(up_spider, down_spider))

    def __call__(self, pattern_func: Any, replacement_func: Any) -> None:
        # # inline: Traced helper
        def get_traced_gm(func) -> fx.GraphModule:
            return fx.GraphModule(torch.nn.Module(), self.tracer.trace(func))

        # # inline: Assert logic helpers
        def get_num_placeholders(gm: fx.GraphModule) -> int:
            return len([n for n in gm.graph.nodes if n.op == "placeholder"])

        def get_num_output_args(gm: fx.GraphModule) -> int:
            output_node = next(n for n in gm.graph.nodes if n.op == "output")
            out_args = output_node.args[0]
            if out_args is None:
                return 0
            return len(out_args) if isinstance(out_args, (tuple, list)) else 1

        def get_num_spiders(gm: fx.GraphModule) -> int:
            # $contain (up_spider | down_spider)
            spider_targets = {up_spider, down_spider}
            return len(
                [
                    n
                    for n in gm.graph.nodes
                    if n.op == "call_function" and n.target in spider_targets
                ]
            )

        # Execute Tracing
        p_gm = get_traced_gm(pattern_func)
        r_gm = get_traced_gm(replacement_func)

        # Assert[NumPlaceholders[Traced[$p]] == NumPlaceholders[Traced[$r]]]
        assert get_num_placeholders(p_gm) == get_num_placeholders(
            r_gm
        ), f"Placeholder mismatch: {get_num_placeholders(p_gm)} vs {get_num_placeholders(r_gm)}"

        # Assert[NumOutputArgs[Traced[$p]] == NumOutputArgs[Traced[$r]]]
        assert get_num_output_args(p_gm) == get_num_output_args(
            r_gm
        ), f"Output mismatch: {get_num_output_args(p_gm)} vs {get_num_output_args(r_gm)}"

        # Assert[NumSpiders[Traced[$p], $contain (up_spider | down_spider)] > 0]
        assert (
            get_num_spiders(p_gm) > 0
        ), "AccessTopo validation failed: Pattern function must contain at least one spider op."


# --- Test Suite (Valid and Invalid Cases) ---


def run_tests():
    validator = AccessTopoReductionRuleGraphValidator()

    # 1. VALID CASE: Matches placeholders, outputs, and contains a spider
    def valid_p(x):
        return torch.relu(down_spider(x))

    def valid_r(x):
        return torch.relu(x)

    # 2. INVALID CASE: Placeholder mismatch (2 vs 1)
    def inv_p_args(x, y):
        return down_spider(x) + y

    def inv_r_args(x):
        return x

    # 3. INVALID CASE: Output mismatch (Tuple vs Tensor)
    def inv_p_out(x):
        return down_spider(x), x

    def inv_r_out(x):
        return x

    # 4. INVALID CASE: No Spider in pattern
    def inv_p_no_spider(x):
        return torch.relu(x)

    def inv_r_no_spider(x):
        return x

    print("--- Running AccessTopo Tests ---")

    # Test Valid
    validator(valid_p, valid_r)
    print("[PASS] Valid reduction rule")

    # Test Invalids
    for name, p, r in [
        ("Placeholder Mismatch", inv_p_args, inv_r_args),
        ("Output Mismatch", inv_p_out, inv_r_out),
        ("No Spider in Pattern", inv_p_no_spider, inv_r_no_spider),
    ]:
        try:
            validator(p, r)
            print(f"[FAIL] {name} should have raised AssertionError")
        except AssertionError as e:
            print(f"[PASS] {name} caught: {e}")


if __name__ == "__main__":
    run_tests()
