import torch
import torch.fx
from typing import Callable, Generator, Optional, Any
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher

# --- Viba Design (The Protocol) ---
"""
MatchResult := 
    Object 
    * $anchors [fx.Node] 
    * $node_map dict[$pattern fx.Node, $target fx.Node]

get_all_matched := 
    Generator[MatchResult] 
    <- $target fx.GraphModule 
    <- $pattern fx.GraphModule
    <- $validate (bool <- MatchResult)

get_first_matched := 
    (MatchResult | None) 
    <- $target fx.GraphModule 
    <- $pattern fx.GraphModule
    <- $get_all_matched (Generator[MatchResult] <- $target * $pattern)
"""

# --- Implementation ---

def get_all_matched(
    target: torch.fx.GraphModule, 
    pattern: torch.fx.GraphModule, 
    validate: Callable[[Any], bool] = lambda _: True
) -> Generator[Any, None, None]:
    """
    Implementation of $get_all_matched.
    Uses SubgraphMatcher to find all structural matches of $pattern within $target.
    """
    matcher = SubgraphMatcher(
        pattern.graph, 
        match_output=False, 
        match_placeholder=True
    )
    
    # Filter results based on the $validate predicate
    yield from (m for m in matcher.match(target.graph) if validate(m))

def get_first_matched(
    target: torch.fx.GraphModule, 
    pattern: torch.fx.GraphModule,
    validate: Callable[[Any], bool] = lambda _: True
) -> Optional[Any]:
    """
    Compact wrapper to retrieve only the first MatchResult.
    Returns 'void' (None) if no match satisfies the $validate condition.
    """
    matches = get_all_matched(target, pattern, validate)
    return next(matches, None)

# --- Verification ---

if __name__ == "__main__":
    # 1. Setup Target GraphModule ($target)
    class BigModel(torch.nn.Module):
        def forward(self, x, y, z):
            a = torch.add(x, y)      
            b = torch.mul(a, z)      
            c = torch.relu(b)        
            return c

    # 2. Setup Pattern GraphModule ($pattern)
    def pattern_func(m, n, p):
        res_add = torch.add(m, n)
        res_mul = torch.mul(res_add, p)
        return res_mul

    big_gm = torch.fx.symbolic_trace(BigModel())
    pattern_gm = torch.fx.symbolic_trace(pattern_func)

    # 3. Execute get_first_matched with a $validate constraint
    # Validation: Ensure all anchor nodes are functional calls (not placeholders/outputs)
    result = get_first_matched(
        target=big_gm, 
        pattern=pattern_gm,
        validate=lambda m: all(n.op == 'call_function' for n in m.anchors)
    )

    # 4. Result Handling
    if result:
        print("Match discovered successfully.")
        # Mapping back to Viba: result.anchors corresponds to $anchors
        print(f"Matched Nodes (Anchors): {[n.name for n in result.anchors]}")
        # result.nodes_map corresponds to $node_map
        print(f"Total nodes in mapping: {len(result.nodes_map)}")
    else:
        print("No valid match found.")
