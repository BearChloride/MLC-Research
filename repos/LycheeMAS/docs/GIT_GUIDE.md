# LycheeMAS Git 项目管理规范

> 所有协作者必读。部分规则来自一次真实事故（2026-09-04：旧基线 rebase 把未解决冲突标记与 27 万行已删代码推上主线，现场存档 `backup/jay-rebase-20260904`）。配套阅读：[`ONBOARDING.md`](ONBOARDING.md)（入门与开发规范）。

---

## 1. 分支模型与权限

- **主线 `LycheeMASvX`**（当前 `LycheeMASv0.3`）：**只由项目负责人创建与更新**——版本分支的开设、主线的直接 push、历史管理均为负责人专属；其他成员对主线只读。
- **贡献分支命名**：**`<module>_<方法名>`**，module 取接缝/模块名（`build` / `prerun` / `memory` / `processing` / `postrun` / `eval` / `scripts`）。例：`prerun_agentdropout`、`processing_aggagent`、`postrun_attribution`。
- **备份/存档**：`backup/<描述>-<日期>`（事故现场、里程碑快照），由负责人管理。
- 大版本推进（如 v0.4）由负责人开新分支并公告。

## 2. 贡献全流程（fork 工作流）

```bash
# ── 一次性初始化 ──
# ① 在 GitHub 上 fork ecoli-hit/LycheeMAS 到自己名下，然后：
git clone git@github.com:<你的用户名>/LycheeMAS.git && cd LycheeMAS
git remote add upstream git@github.com:ecoli-hit/LycheeMAS.git

# ── 每个功能开始 ──
git fetch upstream
git checkout -b prerun_<你的方法> upstream/LycheeMASv0.3   # 从主线最新处切分支

# ── 开发中随时同步主线 ──
git fetch upstream && git rebase upstream/LycheeMASv0.3    # 冲突解决在本地

# ── 开发完成 ──
make lint && make test && make selfcheck && make demo      # 四绿（见 §4 清单）
git push origin prerun_<你的方法>
# 在 GitHub 上向 ecoli-hit/LycheeMAS 的 LycheeMASvX 主线发起 Pull Request
```

**PR 要求**：

- 一个 PR 一件事（功能 + 测试 + 文档 = 同一件事的三面，可同 PR；无关功能拆开）；
- 描述写清"做了什么 + 为什么 + 与原版的声明偏差"；附三件套输出；
- 复现论文方法必须注明出处（repo + arXiv）、许可证处理（范例 `methods/processing/aggagent/NOTICE.md`）；
- **不直接 push 主线，不代替负责人合并**；review 意见改完 push 同分支即自动更新 PR。

## 3. 提交规范

- 格式：`<type>(<scope>): <一句话中文摘要>`，type ∈ `feat / fix / refactor / test / docs / chore`，scope 用接缝或模块名（`prerun` / `processing` / `eval` / `scripts`…）。
- 正文写"做了什么 + 为什么 + 声明偏差"（风格参照 `git log` 现有提交）。
- 一个提交一件事；rebase 整理提交只在**自己 fork 的分支**上做。

## 4. push 前检查清单（每次，无例外）

```bash
git fetch upstream && git rebase upstream/LycheeMASv0.3   # ① 先同步主线（冲突解决在本地）
make lint && make test && make selfcheck # ② 三件套全绿
make demo                                # ③ 改了执行链路时端到端不回归
git diff --cached | grep -E '^\+.*(<<<<<<<|>>>>>>>)' && echo "冲突标记！" # ④ 自查
git push origin <module>_<方法名>         # ⑤ 推到自己 fork，再开 PR
```

## 5. 五条红线（事故直接来源，逐条对应）

1. **禁止提交冲突标记**：`<<<<<<< / ======= / >>>>>>>` 进库 = 立即返工（事故中 `registry.py` 因此语法损坏）。rebase/merge 后必须重跑三件套再 push。
2. **禁止从过期基线 rebase 后直接 push**：本地基线落后主线超过一次大重构时，先 `git fetch` + 读最新 `docs/DESIGN.md`，确认目录结构没变过。已删除的目录（如曾经的 `layers/`、`src/autogen/`）在旧工作区里"复活"是最典型的信号——**diff 出现成百上千个"新增文件"时停下来检查**。
3. **禁止对共享分支 force-push**（`--force` / `--force-with-lease` 均属之），除非：负责人批准 + 已建 `backup/` 分支 + 通知所有协作者。
4. **禁止大文件/产物/密钥入库**：`runs/`、`*.jsonl`、模型权重、`.env`、`.vscode/`、vendored 整仓——`.gitignore` 已覆盖，不许 `git add -f` 绕过。
5. **协作者的贡献不丢弃**：即使提交有问题，也走"备份 → 提取 → 以原作者署名（`--author`）重做"的流程，不做无备份的覆盖。

## 6. 出事了怎么办

- **push 被拒（non-fast-forward）**：`git fetch upstream && git rebase upstream/LycheeMASv0.3` 解冲突 → 三件套 → 再 push。**不要**用 force 解决。
- **发现远程被污染**：不要急着改。先留档——`git branch backup/<描述>-<日期> <坏提交>` 推远程 + `git bundle create <文件> <相关引用>` 本地备份——再评估"修复提交"还是"tip 重写"（后者需负责人批准，走 §5-3 流程）。
- **本地被远程重写甩开**（例如负责人重写了主线）：`git fetch && git reset --hard origin/LycheeMASv0.3`；本地未推的工作先 `git stash` 或切备份分支保住。
- **PR 冲突了**：在自己分支上 `git rebase upstream/LycheeMASv0.3` 解决后 `git push --force-with-lease origin <自己的分支>`（force 只允许用在**自己 fork 的分支**上）。
- 任何拿不准的操作：先在 `backup/` 分支上演练，或在群里问一句——恢复一个 force-push 的成本远高于问一句的成本。

## 7. 事故复盘范例（2026-09-04，供对照学习）

**发生了什么**：协作者基于五模块重构**之前**的旧工作区 rebase 后直接 push 主线：①README/DESIGN/registry 带着未解决冲突标记入库（registry.py 语法直接损坏）；②已删除的 vendored AutoGen 整仓（1800+ 文件/27 万行）复活；③新代码写进已删除的旧路径。

**怎么修的**（即 §6 流程的实操）：远程 `backup/` 分支 + 本地 bundle 双备份 → 负责人批准后 `--force-with-lease` 把主线退回干净基线 → 从坏提交里**提取全部有效贡献**（aggagent 聚合器、postrun 接缝脚手架），按新架构落位、以原作者署名重做成干净提交 → 通知协作者 `reset --hard` 同步。

**教训入规**：红线 1/2/3/5、§4 清单第④步、以及"贡献分支 + PR"模型本身。
