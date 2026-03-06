import spot

# 解析公式
f  = spot.formula(
    'G(p0 -> F p1) && '        # 如果 p0 发生，最终 p1 会发生
    'F(p2 && X p3) && '        # 最终 p2 发生，并且下一步 p3 发生
    'G(p4 -> (!p5 U p6)) && '  # 全局 p4 -> p5 不发生直到 p6 发生
    'F(p7 || p8) && '           # 最终 p7 或 p8 发生
    '!p9 U p0'                  # p9 不发生直到 p0 发生
)
f  = spot.formula("finally globally ( finally not prop_3 or finally not prop_4 ) 並且 finally globally ( finally prop_1 and finally prop_7 ) 和 finally not prop_8 或 finally not prop_2 或 finally not prop_6 同时 （ finally prop_5 and ( not prop_2 or not prop_6 ) ）")


# 1. 化简公式
# simplified_f = spot.simplify(f)
# print(f"化简后: {simplified_f}")

# 2. 检查可满足性 (正确的方法)
# translate(f) 会将 LTL 转为自动机，is_empty() 检查自动机是否有接受路径
if not spot.translate(f).is_empty():
    print("该任务是理论上可达成的 (Satisfiable)")
else:
    print("该任务存在逻辑冲突 (Unsatisfiable!)")
