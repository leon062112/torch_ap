import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from typing import List, Any


class OutputAsMutInputsTransformer:
    def __init__(self):
        # __init__ <- ()
        pass

    def __call__(
        self,
        target: fx.GraphModule,
        input_dtypes: List[torch.dtype],
        symbolic_input_shapes: List[List[Any]],
    ) -> fx.GraphModule:
        # 1. ($graph_module_with_sole_submodule <- $target)
        gm_with_sub = self._fold_to_sole_submodule(target)

        # 2. ($symbolic_output_shapes <- $target <- ...)
        output_shapes = self._infer_output_shapes(
            target, input_dtypes, symbolic_input_shapes
        )

        # 3. ($inserted_mut_input_nodes <- $gm_with_sub <- ...)
        mut_input_nodes = self._insert_empty_nodes(
            gm_with_sub, output_shapes, input_dtypes
        )

        # 4. ($sole_submodule <- $gm_with_sub)
        sole_sub = next(
            m
            for m in gm_with_sub.modules()
            if isinstance(m, fx.GraphModule) and m is not gm_with_sub
        )

        # 5. ($sole_submodule_with_mut_inputs <- $sole_sub <- $inserted_mut_input_nodes)
        self._add_mut_placeholders_to_submodule(sole_sub, mut_input_nodes)

        # 6. ($sole_submodule_without_outputs <- $sole_submodule_with_mut_inputs)
        self._convert_outputs_to_mut_ops(sole_sub)

        return gm_with_sub

    def _fold_to_sole_submodule(self, target: fx.GraphModule) -> fx.GraphModule:
        """Inline logic: Folds existing graph into a submodule called 'sub'."""
        # 先创建空GM，再添加submodule，避免recompile时graph为空的问题
        new_gm = fx.GraphModule(torch.nn.Module(), fx.Graph())
        new_gm.add_submodule("sub", target)

        # 现在构建graph
        new_graph = new_gm.graph
        placeholder_nodes = []
        for n in target.graph.nodes:
            if n.op == "placeholder":
                new_ph = new_graph.placeholder(n.target)
                placeholder_nodes.append(new_ph)

        # 调用子模块
        sub_call = new_graph.call_module("sub", args=tuple(placeholder_nodes))
        new_graph.output(sub_call)

        new_gm.recompile()
        return new_gm

    def _infer_output_shapes(self, target, dtypes, in_shapes) -> List[List[Any]]:
        try:
            from torch.export import export
        except ImportError:
            # Fallback: heuristic
            return [in_shapes[0].copy() if in_shapes else [128, 64]]

        try:
            # 创建示例输入
            fake_inputs = []
            for dtype, shape in zip(dtypes, in_shapes):
                fake_inputs.append(torch.empty(shape, dtype=dtype))

            # 导出带有 shape info 的 graph
            ep = export(target, tuple(fake_inputs), dynamic_shapes=None)

            # 获取输出 shape
            output_node = ep.graph_module.graph.output_node()
            out_val = output_node.args[0]

            def get_shape(val):
                if isinstance(val, torch.Tensor):
                    return [int(d) for d in val.shape]
                elif isinstance(val, fx.Node):
                    if "tensor_meta" in val.meta:
                        return list(val.meta["tensor_meta"].shape)
                return None

            shape = get_shape(out_val)
            return (
                [shape] if shape else [in_shapes[0].copy() if in_shapes else [128, 64]]
            )

        except Exception:
            return [in_shapes[0].copy() if in_shapes else [128, 64]]

    def _insert_empty_nodes(self, gm, out_shapes, dtypes) -> List[fx.Node]:
        """Inline logic: Insert torch.empty at the beginning of the main graph."""
        first_node = next(iter(gm.graph.nodes))
        inserted = []
        with gm.graph.inserting_before(first_node):
            for shape in out_shapes:
                node = gm.graph.call_function(
                    torch.empty, args=(shape,), kwargs={"dtype": dtypes[0]}
                )
                inserted.append(node)

        # 更新对 'sub' 的调用参数，包含这些新插入的空 Tensor
        sub_node = next(
            n for n in gm.graph.nodes if n.op == "call_module" and n.target == "sub"
        )
        sub_node.args = (*sub_node.args, *inserted)
        gm.recompile()
        return inserted

    def _add_mut_placeholders_to_submodule(
        self, sub_gm: fx.GraphModule, mut_nodes: List[fx.Node]
    ):
        """Inline logic: Add placeholders to the submodule's graph."""
        with sub_gm.graph.inserting_after(None):  # Insert at beginning
            # 找到现有的最后一个 placeholder 之后插入
            last_ph = None
            for n in sub_gm.graph.nodes:
                if n.op == "placeholder":
                    last_ph = n

            with sub_gm.graph.inserting_after(last_ph):
                for i in range(len(mut_nodes)):
                    sub_gm.graph.placeholder(f"mut_input_{i}")
        sub_gm.recompile()

    def _convert_outputs_to_mut_ops(self, sub_gm: fx.GraphModule):
        """Inline logic: Replace return with in-place copy (copy_)."""
        output_node = next(n for n in sub_gm.graph.nodes if n.op == "output")
        mut_placeholders = [
            n
            for n in sub_gm.graph.nodes
            if n.op == "placeholder" and "mut_input_" in n.target
        ]

        # 假设输出是一个 Tensor 或 Tuple[Tensor]
        out_vals = output_node.args[0]
        if not isinstance(out_vals, (tuple, list)):
            out_vals = [out_vals]

        with sub_gm.graph.inserting_before(output_node):
            for val, ph in zip(out_vals, mut_placeholders):
                # 核心转换：使用 copy_ 实现原地赋值
                sub_gm.graph.call_method("copy_", args=(ph, val))

        # 移除返回值 (返回 void)
        output_node.args = (None,)
        sub_gm.recompile()


def test_main():
    # 构造一个简单的计算图: out = x + y
    class SimpleModel(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    model = SimpleModel()
    gm = fx.symbolic_trace(model)

    transformer = OutputAsMutInputsTransformer()
    new_gm = transformer(gm, [torch.float32], [[128, 64]])

    print("--- Transformed Graph Module Code ---")
    print(new_gm.code)
    print("\n--- Sole Submodule Code (Side Effect Version) ---")
    print(new_gm.sub.code)


if __name__ == "__main__":
    test_main()
