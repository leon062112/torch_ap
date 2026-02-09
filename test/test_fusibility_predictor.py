import torch

from torch_ap.matmul_epilogue_fusibility_predictor import (
    create_predictor_from_func,
)


def test_case_1():
    """Test Case 1: Fixed-point reduction (tanh -> relu -> reduction)"""
    print("=" * 60)
    print("Test Case 1: Algebraic Fixed-Point Loop (tanh -> relu -> reduction)")
    print("=" * 60)

    def case_math(x, w):
        return torch.tanh(x**2) + w

    predictor, epilogue_gm, mm_idx = create_predictor_from_func(case_math)

    print("\n--- Epilogue GraphModule ---")
    print(epilogue_gm.code)

    result = predictor(epilogue_gm, mm_idx)
    print(f"\nFusibility Prediction: {result}")
    return result


def test_case_2():
    """Test Case 2: Multi-load US fusion"""
    print("=" * 60)
    print("Test Case 2: Multi-parameter Topology Fusion")
    print("=" * 60)

    def case_topo(x, w1, w2):
        return x + w1 + w2

    predictor, epilogue_gm, mm_idx = create_predictor_from_func(case_topo)

    print("\n--- Epilogue GraphModule ---")
    print(epilogue_gm.code)

    result = predictor(epilogue_gm, mm_idx)
    print(f"\nFusibility Prediction: {result}")
    return result


def test_case_3():
    """Test Case 3: Tuple Output"""
    print("=" * 60)
    print("Test Case 3: Tuple Return Indexing")
    print("=" * 60)

    def case_tuple(x, w):
        return torch.relu(x) + w, x

    predictor, epilogue_gm, mm_idx = create_predictor_from_func(case_tuple)

    print("\n--- Epilogue GraphModule ---")
    print(epilogue_gm.code)

    result = predictor(epilogue_gm, mm_idx)
    print(f"\nFusibility Prediction: {result}")
    return result


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MatmuEpilogueFusibilityPredicator Tests")
    print("=" * 60 + "\n")

    results = []

    results.append(("Case 1: tanh(x**2) + w", test_case_1()))
    results.append(("Case 2: x + w1 + w2", test_case_2()))
    results.append(("Case 3: Tuple Output", test_case_3()))

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name}: {status}")
