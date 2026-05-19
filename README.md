# PEPesc_enhanced

基于 PEPesc 的高时延动态链路增强代理与 Mininet 实验脚本。

这个仓库公开的是一个精简后的代码版本，主要包含三部分内容：

- `pep.py` 及其依赖模块：PEPesc 代理本体，以及面向动态链路的感知、决策与执行增强逻辑。
- `Mininet-scripts/`：四节点实验拓扑、TPROXY 部署脚本、单场景/批量/长时鲁棒性实验脚本。
- 训练与分析脚本：用于整理运行日志、训练 GRU 模型、推理链路状态、统计实验结果并生成图表。

## 项目简介

PEPesc 是一个面向高时延链路的 TCP Performance Enhancing Proxy。这个版本在原始 PEPesc 基础上增加了动态链路增强能力，核心思路包括：

- 保留规则估计作为稳定基线。
- 支持基于 GRU 的在线链路感知修正。
- 支持 `legacy` 与 `adaptive` 两种控制模式。
- 支持通过 Mininet 构造动态带宽/丢包场景，并自动完成运行、采集和分析。

如果只想运行原始 PEPesc，可使用：

- `--senseMode rule`
- `--controlMode legacy`

如果要启用动态增强能力，则需要额外提供可用的 GRU checkpoint，并使用：

- `--senseMode hybrid_gru`
- `--controlMode adaptive`

## 目录结构

```text
.
├─ libstreamc/
│  ├─ 22.04/
│  │  └─ libstreamc.so
│  └─ 24.04/
│     └─ libstreamc.so
├─ Mininet-scripts/
│  ├─ 4_nodes_topo.py
│  ├─ deploy-proxy-on-node-b.sh
│  ├─ deploy-proxy-on-node-c.sh
│  ├─ remove-proxy-on-node-b.sh
│  ├─ remove-proxy-on-node-c.sh
│  ├─ scenario_from_csv.sh
│  ├─ run_single_scene_capture.py
│  ├─ run_scene_batch.py
│  └─ run_robustness_trials.py
├─ pep.py
├─ channel.py
├─ protocol.py
├─ pystreamc.py
├─ adaptive_schema.py
├─ adaptive_ml.py
├─ adaptive_data.py
├─ infer_gru.py
├─ train_gru_incremental.py
├─ prepare_runs_training_data.py
├─ analyze_current_csv.py
├─ analyze_static_case_repeats.py
├─ analyze_static_system_results.py
├─ analyze_dynamic_system_results.py
├─ analyze_robustness_trials.py
├─ analyze_robustness_completion_summary.py
└─ plot_bw_rule_vs_gru_compare.py
```

其中：

- `pep.py` 是代理主程序。
- `channel.py`、`protocol.py`、`pystreamc.py` 是底层通道、协议和 `libstreamc.so` 绑定。
- `adaptive_*`、`infer_gru.py`、`train_gru_incremental.py` 负责动态感知模型和在线增强逻辑。
- `Mininet-scripts/` 负责拓扑搭建、场景注入和实验自动化。
- `analyze_*.py` 负责对运行日志做离线统计和画图。

## 环境要求

推荐环境：

- Ubuntu 22.04 或 24.04
- Python 3.10+
- Mininet
- `iptables` / `iproute2` / `tc`
- `iperf`

常用 Python 依赖：

- `numpy`
- `pandas`
- `matplotlib`
- `torch`（仅在训练或启用 `hybrid_gru` 时需要）

`pep.py` 运行时还依赖 `libstreamc.so`。仓库中已经提供了 Ubuntu 22.04 和 24.04 的预编译版本。

## 快速开始

### 1. 配置动态库

根据系统版本选择对应的 `libstreamc.so`：

```bash
export LD_LIBRARY_PATH=$PWD/libstreamc/24.04:$LD_LIBRARY_PATH
```

如果你使用 Ubuntu 22.04，则改为：

```bash
export LD_LIBRARY_PATH=$PWD/libstreamc/22.04:$LD_LIBRARY_PATH
```

### 2. 启动 Mininet 四节点拓扑

四节点拓扑的地址分配如下：

- `nodeA`: `10.0.0.1`
- `nodeB`: `10.0.0.2` / `10.0.1.2`
- `nodeC`: `10.0.1.3` / `10.0.2.3`
- `nodeD`: `10.0.2.4`

其中 `nodeB <-> nodeC` 为受控瓶颈链路，默认条件为：

- 带宽 `20 Mbps`
- 单向时延 `300 ms`
- 丢包率 `1%`

启动拓扑：

```bash
sudo python3 Mininet-scripts/4_nodes_topo.py
```

进入 Mininet CLI 后打开四个终端：

```bash
xterm nodeA nodeB nodeC nodeD
```

### 3. 在 nodeB / nodeC 上部署透明代理规则

在 `nodeB` 中执行：

```bash
bash Mininet-scripts/deploy-proxy-on-node-b.sh
```

在 `nodeC` 中执行：

```bash
bash Mininet-scripts/deploy-proxy-on-node-c.sh
```

这两条脚本会配置：

- `net.ipv4.ip_forward=1`
- `iptables` TPROXY 规则
- `ip rule`
- `ip route`

如果要清理规则，可分别执行：

```bash
bash Mininet-scripts/remove-proxy-on-node-b.sh
bash Mininet-scripts/remove-proxy-on-node-c.sh
```

### 4. 启动 PEPesc

在 `nodeB` 中运行：

```bash
python3 pep.py \
  --selfIp 10.0.1.2 \
  --selfPort 9999 \
  --peerIp 10.0.1.3 \
  --peerPort 9999 \
  --senseMode rule \
  --controlMode legacy \
  --decisionIntervalMs 50 \
  --detail
```

在 `nodeC` 中运行：

```bash
python3 pep.py \
  --selfIp 10.0.1.3 \
  --selfPort 9999 \
  --peerIp 10.0.1.2 \
  --peerPort 9999 \
  --senseMode rule \
  --controlMode legacy \
  --decisionIntervalMs 50 \
  --detail
```

如果要后台运行，可使用 `nohup` 或你自己的进程管理方式。

### 5. 运行 iperf 测试

在 `nodeD` 中启动服务端：

```bash
iperf -s -p 10000 -i 1
```

在 `nodeA` 中启动客户端：

```bash
iperf -c 10.0.2.4 -p 10000 -i 1 -t 120
```

## 启用动态增强模式

如果你已经有训练好的 GRU checkpoint，可以把运行命令切换为：

```bash
python3 pep.py \
  --selfIp 10.0.1.2 \
  --selfPort 9999 \
  --peerIp 10.0.1.3 \
  --peerPort 9999 \
  --senseMode hybrid_gru \
  --hybridCheckpoint /path/to/checkpoint.pt \
  --hybridDevice cpu \
  --controlMode adaptive \
  --decisionIntervalMs 50 \
  --detail
```

另一端同理，只需把 `selfIp` / `peerIp` 对调。

说明：

- `senseMode=rule`：只使用规则估计。
- `senseMode=hybrid_gru`：在规则基线上叠加 GRU 感知修正。
- `controlMode=legacy`：保持原始 PEPesc 的控制方式。
- `controlMode=adaptive`：启用增强后的决策输出。

## 动态场景实验

### 单场景运行

`Mininet-scripts/run_single_scene_capture.py` 会自动完成以下工作：

- 启动四节点 Mininet 拓扑
- 配置 nodeB / nodeC 的透明代理规则
- 启动两端 PEPesc
- 启动 `iperf`
- 在 `nodeB-eth2` 上按场景脚本动态切换带宽和丢包
- 保存 `scenario.csv`、`link_state.csv`、`iperf_d.log` 等结果

场景文件格式如下：

```csv
bw_mbps,loss_pct,duration_s
22,0.1%,12
18,0.7%,10
14,1.2%,12
20,0.4%,10
26,0.2%,12
16,1.6%,12
24,0.3%,10
```

示例命令：

```bash
sudo python3 Mininet-scripts/run_single_scene_capture.py \
  --scene-name demo_scene \
  --scenario-config ./demo_scene.csv \
  --output-root runs \
  --sense-mode rule \
  --control-mode legacy \
  --decision-interval-ms 50 \
  --overwrite
```

### 批量场景运行

`Mininet-scripts/run_scene_batch.py` 支持从一个 `env.csv` 中读取多个场景。该文件至少需要以下列：

- `scene_id`
- `bw_mbps`
- `loss_pct`
- `duration_s`

可选列：

- `enabled`

示例：

```csv
scene_id,bw_mbps,loss_pct,duration_s,enabled
1,20,0.1%,12,1
,16,1.6%,12,1
2,24,0.3%,10,1
,18,0.7%,10,1
```

运行方式：

```bash
sudo python3 Mininet-scripts/run_scene_batch.py \
  --env-csv ./env.csv \
  --output-root runs \
  --sense-mode rule \
  --control-mode legacy \
  --overwrite
```

### 长时鲁棒性实验

`Mininet-scripts/run_robustness_trials.py` 用于将某个基础场景重复多轮，并统计长时间运行稳定性。

示例：

```bash
sudo python3 Mininet-scripts/run_robustness_trials.py \
  --base-env-csv ./env.csv \
  --scene-id 1 \
  --loops 3 \
  --trials 20 \
  --output-root robustness_runs/original \
  --sense-mode rule \
  --control-mode legacy \
  --overwrite
```

## 训练与离线分析

如果你要使用 `hybrid_gru`，通常需要先准备运行日志并训练模型。

一个典型流程如下：

### 1. 把运行结果整理成训练数据

```bash
python3 prepare_runs_training_data.py \
  --runs-root runs \
  --data-dir data_v2_50ms
```

### 2. 训练或增量训练 GRU

```bash
python3 train_gru_incremental.py -h
```

### 3. 使用训练好的模型做推理

```bash
python3 infer_gru.py -h
```

### 4. 统计实验结果

```bash
python3 analyze_static_system_results.py -h
python3 analyze_dynamic_system_results.py
python3 analyze_robustness_trials.py
python3 analyze_robustness_completion_summary.py -h
```

这些脚本通常会输出：

- 汇总 CSV
- 统计 JSON
- 吞吐 / RTT / queue delay / 恢复时间等图表

## 注意事项

- `pep.py`、Mininet、`iptables`/`tc` 相关脚本通常需要 `root` 权限。
- 默认部署脚本和拓扑脚本使用固定地址：`10.0.0.1 -> 10.0.2.4` 与 `10.0.2.4 -> 10.0.0.1`。如果你修改拓扑，需要同步修改 TPROXY 规则。
- `libstreamc.so` 目前只提供预编译版本，请确保系统版本与目录匹配。

## 引用

如果这个仓库对你的工作有帮助，也建议同时引用原始 PEPesc 论文：

> Ye Li, Liang Chen, Li Su, Kanglian Zhao, Jue Wang, Yongjie Yang, Ning Ge.  
> PEPesc: A TCP Performance Enhancing Proxy for Non-Terrestrial Networks.  
> IEEE Transactions on Mobile Computing, 2023.
