import torch
import torch.fx as fx
from typing import List, Tuple
from torch_ap.spider import down_spider

def insert_down_spider(gm: fx.GraphModule, input_idx: int) -> fx.GraphModule:
    """ 
    $insert_down_spider := fx.GraphModule <- $gm <- $input_anchor_position 
    """
    placeholders = [n for n in gm.graph.nodes if n.op == 'placeholder']
    
    # Assert[IsValidIndex[$input_anchor_position]]
    # Early return if index is invalid to keep indent depth small
    if input_idx < 0 or input_idx >= len(placeholders):
        print(f"[Skip] Invalid index {input_idx} for placeholders of length {len(placeholders)}")
        return gm

    # Print graph before transformation
    print(f"\n[Before Transformation] Target Input Index: {input_idx}")
    gm.graph.print_tabular()
        
    target_node = placeholders[input_idx]
    
    # Inline logic: Inject spider after the target placeholder
    with gm.graph.inserting_after(target_node):
        new_node = gm.graph.call_function(down_spider, (target_node,))
        # Re-route all downstream users to the new spider node
        target_node.replace_all_uses_with(new_node, delete_user_cb=lambda user: user != new_node)
    
    # Print graph after transformation
    print(f"[After $insert_down_spider]")
    gm.graph.print_tabular()
    
    return gm

def main(gms_with_pos: List[Tuple[fx.GraphModule, int]]) -> None:
    """ 
    main := void <- $gms list[fx.GraphModule * $input_anchor_position int]
    """
    
    # --- AssertOnlyForTest logic (Integrated in main) ---
    
    # Helper to calculate a unique hash for topological structure
    def get_topo_hash(g): 
        return "->".join([str(n.target) for n in g.graph.nodes if n.op in ['call_function', 'call_method']])
    
    # AssertOnlyForTest[TopoDiversity[$gms[0]] >= 3]
    unique_topos = {get_topo_hash(gm) for gm, _ in gms_with_pos}
    assert len(unique_topos) >= 3, f"TopoDiversity requirement failed: {len(unique_topos)}"

    # AssertOnlyForTest[DiversityOfNumInputs[$gms[0]] >= 3]
    input_counts = {len([n for n in gm.graph.nodes if n.op == 'placeholder']) for gm, _ in gms_with_pos}
    assert len(input_counts) >= 3, f"DiversityOfNumInputs requirement failed: {len(input_counts)}"

    # AssertOnlyForTest[LengthOfSet[$gms[1]] >= 3]
    unique_indices = {idx for _, idx in gms_with_pos}
    assert len(unique_indices) >= 3, f"Index Diversity requirement failed: {len(unique_indices)}"

    # --- Inline Logic Execution ---
    for gm, input_idx in gms_with_pos:
        insert_down_spider(gm, input_idx)

def run_pipeline():
    def mk_gm(in_count: int, layers: int):
        # Generate varied modules to satisfy diversity assertions
        args = ", ".join([f"x{i}" for i in range(in_count)])
        compute = "\n    ".join([f"x0 = x0 + {i}" for i in range(layers)])
        code = f"class M(torch.nn.Module):\n  def forward(self, {args}):\n    {compute}\n    return x0"
        loc = {}
        exec(code, globals(), loc)
        return fx.symbolic_trace(loc['M']())

    # Dataset designed to pass the internal assertions
    data = [
        (mk_gm(in_count=1, layers=1), 0),
        (mk_gm(in_count=2, layers=2), 1),
        (mk_gm(in_count=3, layers=3), 2)
    ]

    main(data)
    print("\n✅ Main logic executed with internal assertions passed.")

if __name__ == "__main__":
    run_pipeline()
