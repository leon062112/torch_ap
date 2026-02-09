from typing import Callable, List, NamedTuple, Optional, Union
import copy
import torch
import torch.fx as fx

from torch_ap.load_op_inserter_pass import LoadOpInserterPass
from torch_ap.store_op_inserter_pass import StoreOpInserterPass
from torch_ap.pattern_replacer_pass import PatternReplacerPass
from torch_ap.concrete_pass.down_spider_inserter_pass import DownSpiderInserterPass
from torch_ap.spider import down_spider as DS, up_spider as US
from torch_ap.load_store_op import load, store


# TorchFunction := (torch.Tensor | List[torch.Tensor]) <- VariadicList[torch.Tensor]
TorchFunction = Callable[..., Union[torch.Tensor, List[torch.Tensor]]]


class AccessTopoRule(NamedTuple):
    pattern_name: str
    replacement_name: str
    pattern_func: TorchFunction
    replacement_func: TorchFunction


class ConfirmPattern(NamedTuple):
    pattern_name: str
    pattern_func: TorchFunction


class MatmuEpilogueFusibilityPredicator:
    """
    MatmuEpilogueFusibilityPredicator :=
        # __call__
        bool <- $mm_epi fx.GraphModule <- $mm_out_as_epi_in_index int
        # __init__
        <- $config_pattern_rewriters (list[AccessTopoRule] <- list[AccessTopoRule])
        <- $config_pattern_removers (list[ConfirmPattern] <- list[ConfirmPattern])
    """

    def __init__(
        self,
        config_pattern_rewriters: Callable[[List[AccessTopoRule]], List[AccessTopoRule]],
        config_pattern_removers: Callable[[List[ConfirmPattern]], List[ConfirmPattern]],
    ):
        self._config_pattern_rewriters = config_pattern_rewriters
        self._config_pattern_removers = config_pattern_removers

    @property
    def config_pattern_rewriters(self) -> List[AccessTopoRule]:
        return self._config_pattern_rewriters(self._default_rewriters())

    @property
    def config_pattern_removers(self) -> List[ConfirmPattern]:
        return self._config_pattern_removers(self._default_removers())

    def _default_rewriters(self) -> List[AccessTopoRule]:
        return [
            AccessTopoRule("y=x**2", "y=relu(x)", lambda x: x**2, lambda x: torch.relu(x)),
            AccessTopoRule("y=tanh(x)", "y=relu(x)", lambda x: torch.tanh(x), lambda x: torch.relu(x)),
            AccessTopoRule("z=DS(x)+y", "z=DS(US(x,y))", lambda x, y: DS(x) + y, lambda x, y: DS(US(x, y))),
            AccessTopoRule("z=y+DS(x)", "z=DS(US(x,y))", lambda x, y: y + DS(x), lambda x, y: DS(US(x, y))),
            AccessTopoRule("z=relu(DS(x))", "z=DS(x)", lambda x: torch.relu(DS(x)), lambda x: DS(x)),
        ]

    def _default_removers(self) -> List[ConfirmPattern]:
        return []

    def __call__(
        self,
        mm_epi: fx.GraphModule,
        mm_out_as_epi_in_index: int
    ) -> bool:
        """
        Predict if the matmul epilogue is fusible.

        Logic: If all confirm patterns can be successfully removed,
        the graph is fusible.
        """
        working_gm = copy.deepcopy(mm_epi)

        # Step 1: Insert DS/Load/Store
        working_gm = self._insert_spiders(working_gm, mm_out_as_epi_in_index)

        # Step 2: Apply rewriters in fixed-point loop
        working_gm = self._fixed_point_loop(working_gm)

        # Step 3: Check confirm patterns by attempting to remove them
        # Use subgraph_rewriter with identity replacement
        from torch.fx import subgraph_rewriter
        from torch_ap.torch_ap_trace import torch_ap_trace

        for remover in self.config_pattern_removers:
            pattern_gm = torch_ap_trace(remover.pattern_func)
            replacement_gm = torch_ap_trace(remover.pattern_func)  # Identity replacement

            matches = subgraph_rewriter.replace_pattern(
                working_gm, pattern_gm, replacement_gm
            )

            if len(matches) == 0:
                # Pattern could not be removed - not fusible
                return False

        # All patterns removed - fusible
        return True

    def _insert_spiders(self, gm: fx.GraphModule, mm_out_idx: int) -> fx.GraphModule:
        res = DownSpiderInserterPass(input_idx=mm_out_idx)(gm)
        gm = res.graph_module
        res = LoadOpInserterPass()(gm)
        gm = res.graph_module
        res = StoreOpInserterPass()(gm)
        return res.graph_module

    def _fixed_point_loop(self, gm: fx.GraphModule) -> fx.GraphModule:
        curr_gm = gm
        while True:
            any_mod = False
            for rule in self.config_pattern_rewriters:
                replacer = PatternReplacerPass(rule.pattern_func, rule.replacement_func)
                res = replacer(curr_gm)
                if res.modified:
                    any_mod = True
                    curr_gm = res.graph_module
            if not any_mod:
                break
        return curr_gm


def get_output_indices(epilogue_gm: fx.GraphModule) -> List[Optional[int]]:
    out_node = next(n for n in epilogue_gm.graph.nodes if n.op == "output")
    res = out_node.args[0]
    if isinstance(res, (tuple, list)):
        return list(range(len(res)))
    return [None]


def create_epilogue_fusibility_predictor(
    epilogue_gm: fx.GraphModule,
    mm_out_as_epi_in_index: int,
    arg_list: Optional[List[tuple[str, bool]]] = None
) -> tuple[
    Callable[[List[AccessTopoRule]], List[AccessTopoRule]],
    Callable[[List[ConfirmPattern]], List[ConfirmPattern]]
]:
    """
    Create config transformers for FusibilityPredicator.

    Args:
        epilogue_gm: The extracted epilogue GraphModule
        mm_out_as_epi_in_index: Index of matmul output in epilogue inputs
        arg_list: List of (name, is_mm_output) tuples

    Returns:
        Tuple of (config_rewriters, config_removers) transformer functions
    """
    output_indices = get_output_indices(epilogue_gm)

    gm_arg_names = [n.name for n in epilogue_gm.graph.nodes if n.op == "placeholder"]

    if arg_list is None:
        arg_list = [(name, False) for name in gm_arg_names]

    mm_out_arg_names = [name for name, is_mm in arg_list if is_mm]
    other_arg_names = [name for name, is_mm in arg_list if not is_mm]

    def config_rewriters(rules: List[AccessTopoRule]) -> List[AccessTopoRule]:
        return rules

    def config_removers(rules: List[ConfirmPattern]) -> List[ConfirmPattern]:
        removers: List[ConfirmPattern] = []

        for out_idx in output_indices:
            removers.append(ConfirmPattern(
                f"store(DS(x, {out_idx}))",
                lambda x, idx=out_idx: store(DS(x), idx)
            ))

        for arg_name in other_arg_names:
            removers.append(ConfirmPattern(
                f"US(x, load(y, {arg_name}))",
                lambda x, y, n=arg_name: US(x, load(y, n))
            ))

        if mm_out_arg_names:
            mm_arg_name = mm_out_arg_names[0]
            removers.append(ConfirmPattern(
                f"load(x, {mm_arg_name})",
                lambda x, n=mm_arg_name: load(x, n)
            ))

        return removers

    return config_rewriters, config_removers


def create_predictor_from_func(
    epilogue_func: Callable,
    other_arg_names: Optional[List[str]] = None
) -> tuple[MatmuEpilogueFusibilityPredicator, fx.GraphModule, int]:
    """
    Create a FusibilityPredicator from an epilogue function.

    This is a convenience function that:
    1. Traces and processes the epilogue function
    2. Extracts the epilogue GraphModule
    3. Creates the predictor with appropriate config

    Args:
        epilogue_func: The epilogue function to analyze
        other_arg_names: Names of non-mm arguments (optional)

    Returns:
        Tuple of (predictor, epilogue_gm, mm_out_idx)
    """
    from torch_ap.torch_ap_trace import torch_ap_trace
    from torch_ap.concrete_pass.demo_matmul_epilogue_replacer_pass import (
        DemoMatmulEpilogueReplacerPass,
    )
    from torch_ap.trivial_ops_folder_pass import TrivialOpsFolderPass
    from torch_ap.concrete_pass.matmul_epilogue_util import (
        get_matmul_epilogue_arg_name_to_is_mm_out,
    )
    from torch_ap.concrete_pass.matmul_epilogue_extractor_pass import (
        MatmulEpilogueExtractorPass,
    )

    # 1. matmul + epilogue
    demo_pass = DemoMatmulEpilogueReplacerPass(epilogue_func)
    gm = torch_ap_trace(epilogue_func)
    res = demo_pass(gm)
    matmul_plus_epilogue = res.graph_module

    # 2. TrivialOpsFolderPass
    res = TrivialOpsFolderPass()(matmul_plus_epilogue)
    matmul_plus_epilogue = res.graph_module

    # 3. Get parameter information
    arg_list = get_matmul_epilogue_arg_name_to_is_mm_out(matmul_plus_epilogue)

    # 4. Get matmul output index
    mm_idx_list = [i for i, (_, is_mm) in enumerate(arg_list) if is_mm]
    if not mm_idx_list:
        raise ValueError("No matmul output found in epilogue arguments")
    mm_idx = mm_idx_list[0]

    # 5. Extract epilogue sub graph
    epilogue_gm = MatmulEpilogueExtractorPass()(matmul_plus_epilogue).graph_module

    # 6. Create predictor
    config_rewriters, config_removers = create_epilogue_fusibility_predictor(
        epilogue_gm, mm_idx, arg_list
    )
    predictor = MatmuEpilogueFusibilityPredicator(config_rewriters, config_removers)

    return predictor, epilogue_gm, mm_idx
