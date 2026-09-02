# Gangda Slurm 16/16 子队列

**日期**：2026-09-02
**状态**：生效
**总队列**：`gangda`
**总资源**：`slurm2-a3nodesetondem-[0-3]`，每节点 8 H100，共 32 GPU

## 决定

把 `gangda` 硬拆为两个不互借的逻辑 subqueue：

| subqueue | branch | nodes | hard limit |
|---|---|---|---:|
| `gangda_exp-protocol-evolve` | `gangda_exp_protocol_evolve` | `slurm2-a3nodesetondem-[0-1]` | 16 GPU |
| `gangda_wma_evolve` | `gangda_wma_evolve` | `slurm2-a3nodesetondem-[2-3]` | 16 GPU |

节点集合不重叠且覆盖四台机器。这里的“硬”由 Slurm `--nodelist` 实现，不依赖
submission 后才写入的 registry 计数。一个 subqueue 空闲时，另一个默认不能借卡；
以后若要改为可借用的 soft quota，必须另写 spec，不能静默改变。

## Registry 契约

`/rmeng_data/robtang/slurm-queue/registry.json` 仍是唯一 ownership authority。
`schema_version` 保持 1，增加向后兼容的 `subqueues` map；每个新 source 可带
`subqueue`。receipt registration 依据 `ownership.branch` 自动选择 subqueue，并拒绝
receipt 显式 subqueue 与 branch mapping 不一致。

旧 receipt 与 terminal history 不重写。没有 subqueue 标签的历史 source 显示为
`unassigned`，但其作业占用哪个节点，就计入哪个 subqueue 的物理 GPU 使用量。

## Submission gate

每条线的 `third_party/PostTrainBench/.env` 必须同时写：

```text
# exp protocol
POST_TRAIN_BENCH_SLURM_SUBQUEUE="gangda_exp-protocol-evolve"
POST_TRAIN_BENCH_SLURM_NODELIST="slurm2-a3nodesetondem-[0-1]"

# WMA
POST_TRAIN_BENCH_SLURM_SUBQUEUE="gangda_wma_evolve"
POST_TRAIN_BENCH_SLURM_NODELIST="slurm2-a3nodesetondem-[2-3]"
```

`awm ptb check` 在发射前验证：

1. subqueue 存在；
2. 当前 branch 属于它；
3. 展开的 nodelist 与 registry 完全一致；
4. 每节点仍声明 `gres/gpu=8`；
5. reservation、partition、root→user 与 ownership registry 的原有门保持生效。

receipt 的 `subqueue` 与 site 配置随 manifest、顶层 commit、PTB commit 一起留存。

## 视图

```bash
gangda-slurm-queue --summary
gangda-slurm-queue --subqueue gangda_exp-protocol-evolve --summary
gangda-slurm-queue --subqueue gangda_wma_evolve --summary
```

总视图继续负责全局 ownership 守卫；subqueue 视图只缩小展示，不削弱
`OWNERSHIP FAIL`。failure/history/show 仍以 receipt 和 job ID 为事实来源。

## 无中断迁移

- 不取消、不 requeue、不迁移任何既有 job。
- 迁移时唯一 active 的 legacy job `89926` 保持在
  `slurm2-a3nodesetondem-0`，直到自然结束；因此 exp-protocol subqueue 在此期间最多
  只有 15 张空闲卡。
- 新提交才受 16/16 nodelist 边界约束。
- 更新代码、registry、两条线的 gitignored `.env` 后，先运行 unit tests 和两边的
  `awm ptb check`，最后重启 snapshot monitor；全程不调用 `sbatch` 或 `scancel`。
