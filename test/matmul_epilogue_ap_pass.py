import torch
import torch.fx as fx

from torch_ap.ap_pass import ApPass
from torch_ap.match_replace_util import MatchContext


class PatternModule(torch.nn.Module):
    def forward(self, x):
        return x


class MatmulEpilogueApPass(ApPass):
    def pattern(self) -> fx.GraphModule:
        class P(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mod = PatternModule()

            def forward(self, x, y):
                return self.mod(torch.matmul(x, y))

        return fx.GraphModule(P(), tracer.trace(P()))

    def constraint(self, match_ctx) -> bool:
        print("[self.get_epilogue_module(match_ctx).code]=========")
        print(self.get_epilogue_module(match_ctx).code)
        print("[self.get_epilogue_module(match_ctx).code]=========")
        return True

    def get_epilogue_module(self, match_ctx) -> fx.GraphModule:
        pattern_module = match_ctx.pattern.mod
        pattern_call_module_node = self.get_mod_node(match_ctx.pattern)
        target_call_module_node = match_ctx.nodes_map[pattern_call_module_node]
        target_module_name = target_call_module_node.target
        return getattr(match_ctx.target, target_module_name)

    def get_mod_node(self, pattern_gm: fx.GraphModule):
        def is_mod_node(node):
            if node.op != "call_module":
                return False
            if node.target != "mod":
                return False
            return True

        for node in pattern_gm.graph.nodes:
            if is_mod_node(node):
                return node
        return None

    def replacement(self, ctx) -> fx.GraphModule:

        class Replacement(torch.nn.Module):
            def forward(self, x, y):
                # replacement contains 0 call_module nodes
                return torch.matmul(x, y) + 1.0

        return fx.symbolic_trace(Replacement())


if __name__ == "__main__":

    class MatmulEpilogue(torch.nn.Module):
        def __init__(self, bias):
            super().__init__()
            self.bias = bias

        def forward(self, x):
            return x - self.bias

    class SimpleTracer(fx.Tracer):

        def __init__(self, leaf_module_classes):
            super().__init__()
            self.leaf_module_classes = leaf_module_classes

        def is_leaf_module(self, m, n):
            return isinstance(m, self.leaf_module_classes)

    tracer = SimpleTracer(leaf_module_classes=(PatternModule))

    # Target: Matmul -> MatmulEpilogue (call_module)
    class TargetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.epi = MatmulEpilogue(2.0)

        def forward(self, a, b):
            return self.epi(torch.matmul(a, b))

    def fold_submodule(gm):
        from torch_ap.trivial_ops_util import get_trivial_ops_ranges, is_trivial_op

        trivial_ops_ranges = get_trivial_ops_ranges(
            gm,
            is_trivial_op=is_trivial_op,
        )
        print(trivial_ops_ranges)
        num_placholders = sum(1 for node in gm.graph.nodes if node.op == "placeholder")
        subgraph_ranges_from_non_placeholder = [
            (start - num_placholders, end - num_placholders)
            for start, end in trivial_ops_ranges
        ]
        from torch_ap.submodule_fold_util import convert_to_submodules_graph

        return convert_to_submodules_graph(
            gm,
            subgraph_ranges_from_non_placeholder=subgraph_ranges_from_non_placeholder,
        )

    t_gm = fx.GraphModule(TargetModel(), tracer.trace(TargetModel()))
    t_gm = fold_submodule(t_gm)
    print("[after fold_submodule]====")
    print(t_gm.code)
    print("[after fold_submodule]====")

    # --- Verification ---
    print("--- Before Transformation ---")
    print(t_gm.code)

    result_gm = MatmulEpilogueApPass()(t_gm).graph_module

    print("\n--- After Transformation ---")
    print(result_gm.code)

    # Final assertion: NumOfFilteredOps[replacement, "call_module"] == 0
    final_mods = [n for n in result_gm.graph.nodes if n.op == "call_module"]
    assert len(final_mods) == 0
    print(f"\nRemaining call_module: {len(final_mods)}")
    print("Success: Transformation consistent with input/output protocol.")
