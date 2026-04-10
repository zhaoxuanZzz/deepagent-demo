"""快速排序算法验证"""


def quick_sort(arr: list) -> list:
    """快速排序算法"""
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)


def verify_quick_sort():
    """验证快速排序的正确性"""
    test_cases = [
        # 原有测试
        ([3, 6, 8, 10, 1, 2, 1], "乱序数组"),
        ([5, 4, 3, 2, 1], "逆序数组"),
        ([1], "单元素"),
        ([], "空数组"),
        ([1, 1, 1, 1], "重复元素"),
        ([9, 8, 7, 6, 5, 4, 3, 2, 1], "倒序"),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], "已排序"),
        ([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5], "随机重复"),
        # 新增测试
        ([-3, -1, -7, -5, -2], "负数"),
        ([3.14, 2.71, 1.41, 1.73, 0.5], "浮点数"),
        ([0, -1, 2, -3, 4, -5, 0], "含零正负混合"),
        ([1000000, 1, 999999, 2], "大整数"),
        ([2, 2, 2, 2, 2], "全部相同"),
        ([1, 2], "最小规模两元素"),
        ([1, 3, 2], "三元素乱序"),
    ]
    
    print("快速排序验证测试")
    print("=" * 50)
    
    all_passed = True
    
    for i, (arr, desc) in enumerate(test_cases, 1):
        original = arr.copy()
        sorted_arr = quick_sort(arr)
        expected = sorted(original)
        
        is_correct = sorted_arr == expected
        status = "✓ 通过" if is_correct else "✗ 失败"
        
        print(f"\n测试 {i}: {desc} - {status}")
        print(f"  输入:   {original}")
        print(f"  输出:   {sorted_arr}")
        
        if not is_correct:
            print(f"  期望:   {expected}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("所有测试通过！✓")
    else:
        print("存在测试失败！✗")
    
    return all_passed


if __name__ == "__main__":
    verify_quick_sort()