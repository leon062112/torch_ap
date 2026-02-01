import torch
import torch.fx as fx


class ApPass:
    def pattern(self) -> fx.GraphModule:
        """$pattern (fx.GraphModule <- ())"""
        raise NotImplementedError

    def constraint(self, match_context) -> bool:
        """$constraint (bool <- MatchContext)"""
        return True

    def replacement(self, match_context) -> fx.GraphModule:
        """$replacement (fx.GraphModule <- MatchContext)"""
        raise NotImplementedError

    def __call__(self, target: fx.GraphModule):
        """$__call__ (PassResult <- $target fx.GraphModule)"""
        # Local import to manage dependencies within the call scope
        from torch_ap.match_replace_util import fx_graph_replace_first_pattern

        # Execute the transformation logic using the utility function
        gm, modified = fx_graph_replace_first_pattern(
            target, self.pattern, self.constraint, self.replacement
        )
        from torch.fx.passes.infra.pass_manager import PassResult

        return PassResult(gm, modified=modified)


if __name__ == "__main__":

    class LinearEpilogue(torch.nn.Module):
        def __init__(self, bias):
            super().__init__()
            self.bias = bias

        def forward(self, x):
            return x + self.bias

    class PatternModule(torch.nn.Module):
        def forward(self, x):
            return x

    class SimpleTracer(fx.Tracer):
        def is_leaf_module(self, m, n):
            return isinstance(m, (LinearEpilogue, PatternModule))

    tracer = SimpleTracer()

    # Target: Matmul -> LinearEpilogue (call_module)
    class TargetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.epi = LinearEpilogue(2.0)

        def forward(self, a, b):
            return self.epi(torch.matmul(a, b))

    t_gm = fx.GraphModule(TargetModel(), tracer.trace(TargetModel()))

    # --- Verification ---
    print("--- Before Transformation ---")
    print(t_gm.code)

    class DemoApPass(ApPass):
        def pattern(self) -> fx.GraphModule:
            class P(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mod = PatternModule()

                def forward(self, x, y):
                    return self.mod(torch.matmul(x, y))

            return fx.GraphModule(P(), tracer.trace(P()))

        def replacement(self, ctx) -> fx.GraphModule:
            t_node = next(
                tn for pn, tn in ctx.nodes_map.items() if pn.op == "call_module"
            )
            bias_val = t_gm.get_submodule(t_node.target).bias

            class Replacement(torch.nn.Module):
                def forward(self, x, y):
                    # replacement contains 0 call_module nodes
                    return torch.matmul(x, y) + bias_val

            return fx.symbolic_trace(Replacement())

    result_gm = DemoApPass()(t_gm).graph_module

    print("\n--- After Transformation ---")
    print(result_gm.code)

    # Final assertion: NumOfFilteredOps[replacement, "call_module"] == 0
    final_mods = [n for n in result_gm.graph.nodes if n.op == "call_module"]
    assert len(final_mods) == 0
    print(f"\nRemaining call_module: {len(final_mods)}")
    print("Success: Transformation consistent with input/output protocol.")
