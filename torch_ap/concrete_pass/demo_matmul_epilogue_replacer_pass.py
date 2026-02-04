import torch
import torch.fx as fx
from typing import Any
from torch.fx.passes.infra.pass_manager import PassResult


class DemoMatmulEpilogueReplacerPass:
    """
    DemoMatmulEpilogueReplacerPass :=
        PassResult <- None <- $get_torch_module <- $tracer <- $epilogue_func
    """

    def __init__(self, epilogue_func: Any = None):
        if epilogue_func is None:

            def epilogue_func(x, bias):
                return torch.tanh(x + bias)

        # 1. Capture $epilogue_func
        self.epilogue_func = epilogue_func

        # 2. $tracer fx.Tracer
        self.tracer = fx.Tracer()

        # 3. $get_torch_module (torch.nn.Module <- $epilogue_func)
        # Inline construction of the module to be traced
        def get_torch_module(epi_fn: Any) -> torch.nn.Module:
            class GeneratedModule(torch.nn.Module):
                def forward(self, x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor):
                    # matmul + epilogue (indicated by $epilogue_func)
                    out = torch.matmul(x, y)
                    return epi_fn(out, bias)

            return GeneratedModule()

        self.mod = get_torch_module(self.epilogue_func)

    def __call__(self, _unused: None) -> PassResult:
        """
        Executes the trace on the synthesized module.
        """
        # Symbolic tracing via $tracer
        graph = self.tracer.trace(self.mod)
        generated_gm = fx.GraphModule(self.mod, graph)

        # Always returns modified=True as it generates a new GraphModule
        return PassResult(graph_module=generated_gm, modified=True)


# --- Test Environment ---


def main():
    # Instantiate with the epilogue function
    replacer_pass = DemoMatmulEpilogueReplacerPass()

    # Invoke with None as per Viba signature
    result = replacer_pass(None)

    print("--- [Generated Graph: Matmul + Tanh(Square)] ---")
    result.graph_module.graph.print_tabular()

    print("\n--- [Generated Python Code] ---")
    print(result.graph_module.code)


if __name__ == "__main__":
    main()
