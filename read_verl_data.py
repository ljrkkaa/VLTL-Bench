import pandas as pd

# 读取 parquet 文件
df = pd.read_parquet("VLTL-Bench/new_generated_datasets/warehouse_nl.parquet")

# 查看数据表的前几行
print(df.head(1))  # 第一组数据

# 如果想看更多信息，比如列名
print(df.columns)