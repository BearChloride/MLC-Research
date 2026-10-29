# `memory` — 记忆层（运行时组件，顶层包）

## 功能

回答「**智能体之间记住什么、以什么表征传递**」。两个对称的可替换接缝：

- **方法接缝（manager，类别 `memory_manager`）**：记忆如何存储/召回/物化——`managers/`。
- **触发接缝（router，类别 `memory_router`）**：当前 agent 本轮用哪个通道——`routing/`。

共享注入引擎（`runtime/injection.py`）在每次 agent 发言时依次调 router → manager，把产出的 `MemoryBundle` 注入生成过程。消费方只认 `MemoryBundle` / `RouteDecision`，换 manager/router 即一组对照实验。

## 接口

### `base.py` — 方法接缝契约

```python
@dataclass
class MemoryBundle:                     # 本轮要注入的内容（每通道一对「内容 + 产出方法」）
    NL_Channel: Optional[str]           # 文本通道注入内容
    NL_strategy: Optional[str]          # 产出它的 NL 方法名
    Latent_Channel: Optional[Any]       # 隐空间载荷（张量 / projector 栈）
    Latent_strategy: Optional[str]      # 产出它的 latent 方法名（决定注入方式）
    meta: dict

class MemoryManager(ABC):
    name: str
    def observe(self, messages: list[Any]) -> None                        # 更新记忆库
    def recall(self, decision: RouteDecision, query: str) -> MemoryBundle # 物化选定通道
    def reset(self) -> None                                               # 清空单次对话状态
```

### `routing/base.py` — 触发接缝契约

```python
Channel = Literal["none", "nl", "latent", "both"]

@dataclass
class RouterInputs:      # 路由器可条件化的信息
    role / task / turn / sender / receiver / query / availability / same_model_pair

@dataclass
class RouteDecision:
    channel: Channel; reason: str      # uses_latent() / uses_nl()

class MemoryRouter(ABC):
    def decide(self, x: RouterInputs) -> RouteDecision: ...
```

### 其它模块

- **`context.py` — `RoutingContext`**：跨 agent 共享的路由/记账状态（turn 计数、role→model 表、决策日志 `decisions`、可选 TraceStore / JsonlSpanLogger）。接口：`turn_of / bump_turn / same_model_pair / log_decision / set_case / log_span / reset`。
- **`store.py` — `MemoryStore`**：key→value 缓存接缝（可选容量 + FIFO）：`put/get/has/clear/__len__`。
- **`channels/`**：表征通道实现——`nl.py`（文本通道，多种 strategy）与 `latent.py`（隐空间通道统一接口，多种物化策略）；`c2c_*.py` 是 latent 的内部子实现，manager 不直接依赖。

## 已注册组件

| 类别 | 已实现 | 桩 |
|---|---|---|
| `memory_manager` | `cdm`（双通道） | `mem0`, `ama` |
| `memory_router` | `static`（(role×task)→channel 策略表）, `fixed`（恒定通道；工厂 `fixed_channel_router`） | `learned`, `soft_gate` |

## 约定

- import 本包触发全部注册，不触发 torch（惰性导入）。**`channels/c2c_projector.py` 顶层 import torch，不得出现在注册链上**（只被惰性加载函数引用）。
- 新增记忆方法/路由算法：继承对应 ABC + 注册 + 子包 `__init__` 触发 + config + test（六步配方）。
