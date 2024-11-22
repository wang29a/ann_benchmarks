# 在 OceanBase 上运行 ann-benchmark

0. 安装所需 python 依赖，并下载数据集
```
pip install -r requirements.txt
wget -O data/sift-128-euclidean.hdf5 http://ann-benchmarks.com/sift-128-euclidean.hdf5
```
1. 编译/部署 OceanBase 数据库，并创建用于测试的租户
```bash
# 编译
bash build.sh release -DOB_USE_CCACHE=ON --init --make
# 部署
# 在 oceanbase 目录下运行
./tools/deploy/obd.sh prepare -p /tmp/obtest
# 配置文件请参考下面 obcluster.yaml 示例
./tools/deploy/obd.sh deploy -c obcluster.yaml
```
2. 运行 ann-benchmark.
```bash
# 测试时导入数据并构建索引
python run.py --algorithm oceanbase --local --force --dataset sift-128-euclidean --runs 1
# 测试时跳过导入数据及构建索引
python run.py --algorithm oceanbase --local --force --dataset sift-128-euclidean --runs 1 --skip_fit
# 计算召回率及 QPS
python plot.py --dataset sift-128-euclidean --recompute
```
3. 生成运行结果.
```bash
# 示例命令
python plot.py --dataset sift-128-euclidean --recompute
# 示例输出如下，其中每行结果倒数第一个值为该算法对应的QPS，每行结果倒数第二个值为该算法对应的召回率。
Computing knn metrics
  0:                               OBVector(m=16, ef_construction=200, ef_search=400)        0.999      416.990
Computing knn metrics
  1:                                                                 BruteForceBLAS()        1.000      355.359
```

## 运行向量标量混合场景测试
```bash
python -m ann_benchmarks.algorithms.oceanbase.hybrid_ann
python -m ann_benchmarks.algorithms.oceanbase.hybrid_ann --skip_fit
```