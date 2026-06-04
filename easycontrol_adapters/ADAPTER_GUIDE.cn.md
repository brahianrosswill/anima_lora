# 构建你自己的 EasyControl 适配器

本指南手把手讲解如何为 Anima 添加一个**新的 EasyControl 控制任务**，供你自己使用——不是作为 git 贡献，只是一个本地适配器，存放在
`easycontrol_adapters/<your_task>/` 下。我们以规范示例 **colorize**
（`easycontrol_adapters/colorization/`）为例详细讲解，并明确指出针对你自己的任务需要修改哪些地方。

如果你只读一句话：EasyControl 适配器**不是新的模型代码**。网络
（`networks/methods/easycontrol.py`）、双流前向、`b_cond` logit 偏置、推理 KV 缓存——
这些都已随仓库提供并共享。一个控制任务与其他任何控制任务的区别*仅在一个*维度上：

> **条件图像是如何构建的。**

下文的所有内容都只是围绕这一个核心想法的管道搭建。

---

## 0. 心智模型——EasyControl 适配器究竟是什么

EasyControl 以一张参考图像作为生成的条件。该参考图像被 VAE 编码成 *cond token*，这些
token 与目标流并行流经每一个 DiT 块；目标的自注意力会关注一个扩展的 key 集合
`[target_k; cond_k]`。（完整架构见 `docs/experimental/easycontrol.md`。）

**默认**的 EasyControl 使用 `cond == target`（参考图像*就是*正在被重建的图像）。
**控制任务适配器**打破了这一点：它将每张彩色目标与一张*不同的*条件图像配对，从而让模型学习
`condition → target` 而不是恒等映射。

| | 默认 EasyControl | 控制任务适配器（如 colorize） |
|---|---|---|
| target | 图像 X | 彩色图像 X |
| condition | 图像 X（同一个潜在变量） | X 的一个*变换*（X 的黑白漫画版） |
| 它学到什么 | 重建 | manga → color |
| 文本通道 | 完整标注 | （可选）精简标注 |

colorize 的洞见，你可以复用：**真实的黑白漫画没有彩色的真值（ground truth）**，所以你无法
通过采集来构造 `(B&W, color)` 配对。你要*反转*方向——把你手头已有的彩色图像作为目标，
然后**合成**出每张图对应的黑白条件（XDoG 线条 + 算法生成的网点）。让合成出的条件匹配推理时的
分布，这才是整件事的关键。如果你要构建的是,比如说，`depth → image` 或 `pose → image`，那么你的
“mangafy”步骤就是在已有训练图像上运行的深度估计器或姿态提取器。

所以你的工作就是：**写一个函数 `color_image → condition_image`，把它的输出缓存为一组
并行的潜在变量，让一个数据集蓝图指向它，再加上一个 config 加上一行选择器条目。**

---

## 1. 你需要改动的四个层面

添加一个名为 `<task>` 的适配器，意味着恰好需要创建/编辑以下这些：

| # | 层面 | colorize 实例 | 作用 |
|---|---------|-------------------|--------------|
| 1 | `easycontrol_adapters/<task>/` 项目 | `colorization/`（`mangafy*.py`、`color_caption.py`、`prep.py`） | 构建并缓存条件（以及可选的精简文本缓存） |
| 2 | `configs/datasets/<task>.toml` | `configs/datasets/colorize.toml` | 数据集蓝图，将目标潜在变量与 **cond_cache_dir** 配对 |
| 3 | `configs/methods/<task>.toml`（+ `configs/gui-methods/<task>.toml`） | `configs/methods/colorize.toml` | 方法 config——指向数据集，设置 LR/epochs/`network_args` |
| 4 | `scripts/experimental_tasks/{training,inference}.py` 选择器 | `_EASYADAPTERS = {"colorize"}` + 分支 | 把 `EASYADAPTER=<task>` 接入 `make exp-easycontrol*` 目标 |

我们按顺序逐一讲解。这里没有任何内容需要改动 `networks/`。

---

## 2. 层面 1——适配器项目（`easycontrol_adapters/<task>/`）

真正的工作在这里。它有两个职责：**合成条件**和**缓存它**（外加可选地构建一个任务专属的文本缓存）。

### 2a. 条件合成器

一个纯函数：彩色 RGB `uint8 (H,W,3)` + 一个按 stem 划分的种子 → 条件 RGB
`uint8 (H,W,3)`，尺寸**相同**。（尺寸相同很重要——见 §3 关于 token 数匹配的说明。）给定种子时它必须是
确定性的，这样重跑和并行 worker 才能逐位一致。

在 colorize 中这是 `mangafy.py::mangafy_array`（及其 CUDA 孪生体
`mangafy_gpu.py::mangafy_array_gpu`）：

```python
# easycontrol_adapters/colorization/prep.py
Screener = Callable[[np.ndarray, int], np.ndarray]  # (img_rgb, seed) → cond_rgb
```

值得照搬的关键设计点：

- **按 stem 的确定性种子。** colorize 使用 `zlib.crc32(stem)`——
  *而非* Python 的 `hash()`，后者按进程加盐，会让各个 worker 结果不一致。种子抖动
  （colorize 中按页的网点角度/周期）能在不引入非确定性的前提下带来多样性。
- **延迟的重量级导入。** colorize 有三种引擎（`cv2` / `gpu` / `sd`），只有当某个页面
  实际路由到 SD 时才导入那个 3.5 GB 的 SD 栈。如果你的合成器需要一个模型（深度网络、
  线条提取器），就惰性导入它，让无需下载的兜底路径保持轻量。
- **无模型兜底是宝贵的。** colorize 的 `cv2`/`gpu` 引擎不需要任何下载。如果你的任务也有
  这样的兜底，你就能在一个全新检出的仓库上直接做 prep + 训练，无需 `make exp-easycontrol-download`
  这一步。
- **原子写入。** colorize 的 `_save_png_atomic` 先写临时文件再 `os.replace`，这样一次被中断的
  运行永远不会留下被截断的 PNG，否则 `out.exists()` 跳过检查会永远信任它。照搬这个；否则这是
  你迟早会撞上的真实 bug。

如果你的条件是一个*真实*的工件（你已经在磁盘上有深度图/草图），你完全可以跳过合成，直接让
编码阶段指向它们。只有当你必须从目标派生出条件时，才需要合成。

### 2b.（可选）精简的文本缓存

colorize 不仅改变条件——它还**把标注精简为仅颜色标签**
（`color_caption.py::filter_to_colors`）。其中的道理是任务专属且很有启发性的：条件
（线条 + 网点）已经编码了*所有空间信息*，所以留给文本去承载的只剩黑白图无法表达的那个变量——
**色调（hue）**。把标注过滤成颜色标签，使得每个幸存下来的 token 都是模型无法从结构中获取的事实，
从而得到一个强的 `prompt → color` 绑定，而不是把颜色标签埋在完整标注里产生的弱引导。

为你的任务问同样的问题：**条件已经确定了什么，又留下什么对文本而言是模糊的？** `pose → image`
条件固定了姿态但没固定外观/服装/场景——所以你大概会保留*完整*标注。`depth → image`
条件固定了布局但没固定身份或调色板。colorize 那个“把标注过滤到残余模糊量”的做法是一个*模式*，
不是硬性要求；许多适配器原样保留标注，完全跳过文本缓存（只需从数据集蓝图中省略
`text_cache_dir`——§4）。

如果你确实要精简标注，注意 colorize 的两个相互独立的旋钮（别把它们混为一谈——README
在这一点上花了不少笔墨）：

- **`caption_dropout_rate`**——自动上色的*下限（floor）*。约 5% 的训练步会完全丢弃
  标注（→ uncond），训练空提示下的默认行为。保持它**低**
  （`0.05`）；过高的比率会把无条件路径过度训练成弱引导。
- **`use_shuffled_caption_variants`**——完整 vs 部分的*平衡*。文本缓存是多变体的
  （v0 = 完整颜色集合，v1+ = 每个标签以 p≈0.5 丢弃后的打乱版），加载器按 20% v0 / 80% v1+ 抽取，
  这样部分提示（单独的 "pink hair"）也能生效。

### 2c. `prep.py`——缓存构建器

三个**幂等**阶段（跳过已完成的工作；可安全重跑）：

1. **合成（Synthesize）**——遍历 `--src`
   （`post_image_dataset/resized`）下的每一张彩色图，运行合成器，把条件 PNG 写入一个镜像源
   子路径的 `--staging` 目录。
2. **编码（Encode）**——通过 `library.preprocess.cache_latents` 把暂存的条件图像 VAE 编码进
   `--cond_cache_dir`，按每张图的**原生尺寸**编码，使 cond 潜在变量的形状与其目标潜在变量精确一致。
   与目标缓存采用相同的 `{stem}_{WxH}_anima.npz` 格式。
3. **（可选）文本（Text）**——通过 `library.preprocess.cache_text_embeddings`，带上一个
   `caption_transform=`（以及 `caption_shuffle_variants` / `caption_tag_dropout_rate`），
   把标注经过你的过滤器重新编码进一个 `--text_cache_dir`。

复用 library 原语——`library.preprocess.{cache_latents,
cache_text_embeddings, tqdm_progress}` 和 `library.preprocess._dataset.walk_images`。
不要手搓编码循环；`prep.py` 是它们之上的一层薄薄的编排外壳，正如 `scripts/preprocess/*.py` 那样。

colorize 处理了两个正确性陷阱，你照搬它的结构就能免费继承：

- **Stem 键匹配。** 文本阶段从标注主目录读取 `.txt`
  （`image_dataset/`，其嵌套结构与 `resized/` 完全相同），使得生成的 TE 缓存路径能与加载器的
  `image_dir=post_image_dataset/resized` 查找做键匹配。如果你的缓存不能与目标 stem 做键匹配，
  加载器会静默地不去配对它们。
- **Uncond 旁挂文件（sidecar）。** colorize 的文本阶段幂等地重新暂存共享的 `T5("")` uncond
  旁挂文件，以防 colorize 这次运行是第一个触及它的。如果你构建了文本缓存并使用了 caption dropout，
  也要这么做（`library.inference.uncond.stage_uncond_sidecar_with_models`）。

---

## 3. 关键不变量——cond token 数必须匹配目标的 token 数

DiT 运行在 Anima 的**原生形状 token 桶（native-shape bucketing）**上（两个 token 数家族，
4032 / 4200；见 CLAUDE.md 和 `docs/experimental/easycontrol.md` 的 §"Cond token
count"）。**不存在静态填充（static-pad）旋钮**——cond 流以 cond 潜在变量的原生 token 数运行。

这正是为什么 §2a 坚持合成器的输出与输入**尺寸相同**，而 §2c 按**原生尺寸**编码：这样条件潜在变量
就会自动落在与其目标潜在变量相同的桶家族里，`_extended_target_attention` 也就直接能用。如果你要
下采样条件（这是合理的——更小的 cond = 更省内存 / 更快），请在图像层面*上游*进行，让编码出的潜在
变量仍落在一个真实的桶上；不要试图在网络里限制 token 数。

---

## 4. 层面 2——数据集蓝图（`configs/datasets/<task>.toml`）

这就是让 `cond ≠ target` 成立的东西。它是一个普通的数据集蓝图（`[general]`
+ `[[datasets]]` + `[[datasets.subsets]]`），只多了**一个额外的 subset 旋钮**：
`cond_cache_dir`（以及可选的 `text_cache_dir`）。

colorize（`configs/datasets/colorize.toml`），带注释：

```toml
[general]
caption_extension = '.txt'
keep_tokens = 3

[[datasets]]
batch_size = 1
validation_split = 0.005
validation_seed = 42

  [[datasets.subsets]]
  image_dir = 'post_image_dataset/resized'        # the COLOR targets
  cache_dir = 'post_image_dataset/lora'           # target latents+TE — REUSED, not re-encoded
  cond_cache_dir = 'post_image_dataset/colorize_cond'   # ← the synthetic condition latents (prep.py)
  text_cache_dir = 'post_image_dataset/colorize_text'   # ← color-only TE cache (prep.py); TE-only redirect
  recursive = true
  flip_aug = false        # latents can't be flipped post-hoc, and the cond cache has no flipped variant
  num_repeats = 1
```

每个重定向的作用：

- **`cond_cache_dir`**——唯一区分控制任务适配器的旋钮。加载器在这里把每个目标做 stem 匹配到一个
  条件潜在变量。这正是 EasyControl 的双流前向作为参考所消费的东西。
- **`text_cache_dir`**——一个 **TE-only** 重定向（潜在变量仍来自
  `cache_dir`）。如果你保留完整标注就整个省略它——那样加载器会读取共享的 TE 缓存，你也就跳过了
  `prep.py` 的文本阶段。
- **`flip_aug = false`**——必需。翻转后的目标会需要一个翻转后的条件潜在变量，而你并没有缓存它。
  保持翻转关闭。

注意 colorize 复用**共享的** `post_image_dataset/lora` 缓存作为目标潜在变量和 TE——
`make preprocess` 已经构建好了它们，没有任何东西被重新编码。你的适配器只额外增加*条件*缓存
（以及可能的精简文本缓存）。

---

## 5. 层面 3——方法 config（`configs/methods/<task>.toml`）

它几乎是 `configs/methods/easycontrol.toml` 的克隆。唯一的结构性改动是
`dataset_config` 指向你的蓝图；其余都是超参数。

colorize（`configs/methods/colorize.toml`），承重的几行：

```toml
dataset_config = "configs/datasets/colorize.toml"   # ← your blueprint from §4

network_module = "networks.methods.easycontrol"     # SHARED — same network as default

network_dim = 32
network_alpha = 32
network_args = [
    "b_cond_init=-6.0",     # logit-bias init (see below)
    "cond_scale=1.0",
    "apply_ffn_lora=1",     # 0 → drop FFN LoRA, ~halves trainable params
]

output_name = "anima_colorize_full"   # ← checkpoint name; the inference selector greps this

use_easycontrol = true
easycontrol_drop_p = 0.0              # image-CFG dropout; 0.1 default, colorize wants 0
masked_loss = false

caption_dropout_rate = 0.05           # auto-color floor (§2b)
use_shuffled_caption_variants = true  # full-vs-partial balance (§2b)
easycontrol_cond_noise_max = 0.02     # small — high noise erases the lineart structure

learning_rate = 2e-5
max_train_epochs = 3
blocks_to_swap = 0                    # recommended for EasyControl
gradient_checkpointing = true
unsloth_offload_checkpointing = true
```

为你的任务需要斟酌的旋钮：

- **`b_cond_init`**——step-0 基线等价初始化。`-10` 使 cond 在 step 0 时只贡献 softmax 质量的约
  `e⁻¹⁰`（= 基线 DiT，然后逐渐学高）；`docs/experimental/easycontrol.md` 的 §"Step-0 baseline
  equivalence" 推导了这一点。colorize 把它放宽到 `-6`，让 cond 更早起作用——一个条件更强的任务
  可以承受这一点。无论哪种情况它都是可学习的。
- **`easycontrol_cond_noise_max`**——施加在 cond 潜在变量上的逐步训练噪声
  （σ ~ U(0, max)，以 `cond + σ·ε` 形式施加）。`0` = cond 是一个完美的蓝图；正值会把它退化为
  “有损提示”，迫使文本去承载残余细节。colorize 使用 `0.02`（极小——线条*就是*信号本身）。默认的
  easycontrol.toml 使用 `0.3`。
- **`easycontrol_drop_p`**——为 image-CFG 设置的逐 batch 全 cond dropout。colorize
  设为 `0`（条件总是需要的）；默认是 `0.1`。
- **`output_name`**——必须是一个唯一的 stem；推理选择器据此解析出最新的检查点（§6）。

可选地也添加 `configs/gui-methods/<task>.toml`——一个自包含的变体（无 toggle 块），带一个
`[variant]` 块（`family = "easycontrol"`、`label`、`description`、`order`），这样它就会出现在
GUI 的 EasyControl 标签页下拉菜单中。参见
`configs/gui-methods/colorize.toml`。如果你只从 CLI 运行就跳过这一步。

---

## 6. 层面 4——把 `EASYADAPTER=<task>` 接入任务运行器

`make exp-easycontrol*` 目标根据 `EASYADAPTER` 环境变量进行分发。三处小改动让
`EASYADAPTER=<task>` 路由到你的 config + prep + 检查点。

**`scripts/experimental_tasks/training.py`：**

1. 把名字注册进允许列表：
   ```python
   _EASYADAPTERS = {"colorize", "<task>"}   # was {"colorize"}
   ```
   （`_easyadapter()` 会校验这个集合，遇到拼写错误就报错。）

2. 在 `cmd_easycontrol_preprocess` 中把预处理路由到你的 `prep.py`：
   ```python
   adapter = _easyadapter()
   if adapter == "colorize":
       run([PY, "easycontrol_adapters/colorization/prep.py", *extra]); return
   if adapter == "<task>":
       run([PY, "easycontrol_adapters/<task>/prep.py", *extra]); return
   ```

3. 训练本身**无需**改动——`cmd_easycontrol` 已经做了
   `train(_easyadapter() or "easycontrol", extra)`，所以只要名字进了允许列表，`EASYADAPTER=<task>`
   就会自动运行 `configs/methods/<task>.toml`。

4. （仅当你的合成器需要下载时）在 `cmd_easycontrol_download` 中添加一个分支，指向你的权重抓取任务。

**`scripts/experimental_tasks/inference.py`**（`cmd_test_easycontrol`）：该选择器目前把 colorize
硬编码了。把这三个 colorize 专属的值针对你的任务泛化——检查点名、输出子目录、ref 兜底目录，
以及空提示默认行为：

```python
adapter = (os.environ.get("EASYADAPTER") or "").strip()
is_colorize = adapter == "colorize"
weight_name = "anima_colorize" if is_colorize else "anima_easycontrol"
out_sub     = "colorize"       if is_colorize else "easycontrol"
ref_fallback_dir = (ROOT/"post_image_dataset"/"resized") if is_colorize else (ROOT/"easycontrol-dataset")
```

在旁边加上你的 `adapter == "<task>"` 分支（权重名必须匹配你 config 的 `output_name`，对照
`latest_output` 的 stem 匹配）。如果你的任务像 colorize 一样想要空提示默认行为以及来自某个特定兜底
目录的真实条件图像，就照搬下方的 `is_colorize` 分支。

---

## 7. 运行它

```bash
# 1. Build the condition cache (synthesize + VAE-encode). Idempotent.
make exp-easycontrol-preprocess EASYADAPTER=<task>
#    QA a handful first:
python easycontrol_adapters/<task>/prep.py --limit 8
#    Inspect the staged condition PNGs under post_image_dataset/<task>_staging/

# 2. Train (frozen DiT, adapter-only).
make exp-easycontrol EASYADAPTER=<task>

# 3. Inference — feed a real in-distribution condition image as the reference.
REF_IMAGE=path/to/condition.png make exp-test-easycontrol EASYADAPTER=<task>
#    Steer with text:  ... ARGS='--prompt "..."'
```

前置条件：先运行一次 `make preprocess`，使共享的目标潜在变量 + TE 存在于
`post_image_dataset/lora`（你的适配器会复用它们）。

### 值得了解的推理设置（来自 colorize 的经验）

- **喂入一张真实的分布内条件。** 参考图像在推理时被*原样* VAE 编码——不做合成。colorize 喂入
  一张真实的带网点的黑白页面；一张平淡的灰度照片是分布外的，会导致退化。无论你的合成器*模仿*了
  什么，推理时就期望收到那样的东西。
- **`--easycontrol_image_match_size`**——挑选匹配参考图像宽高比的 token 桶，这样高瘦的页面就
  不会被压扁。colorize 强制开启它。
- **`--easycontrol_scale`**（`EC_SCALE=`，结构遵循度）——训练默认为 `1.0`；如果条件渗色就调高
  （1.1–1.2），想要更松散的输出就调低（0.7–0.8）。
- **`--guidance_scale`**——与你的文本策略相互作用。colorize：空提示 → 低 cfg
  （1.0–1.5，没有什么可推向的目标）；文本提示 → 更高（3.0–4.5，正是这让提示起作用）。

---

## 8. 检查清单

- [ ] `easycontrol_adapters/<task>/`，包含一个确定性、原子写入的合成器
      （`(img, seed) → cond`，同尺寸）和一个幂等的 `prep.py`
      （合成 → 编码 → 可选文本）。
- [ ] 条件按**原生尺寸**编码，使 token 数匹配目标桶家族（§3）。
- [ ] `configs/datasets/<task>.toml`，带 `cond_cache_dir`（+ 可选的
      `text_cache_dir`）、`flip_aug = false`，并复用共享的目标缓存。
- [ ] `configs/methods/<task>.toml` → 指向蓝图，唯一的
      `output_name`，`network_module = "networks.methods.easycontrol"`，
      `use_easycontrol = true`。（可选 GUI 变体。）
- [ ] `EASYADAPTER=<task>` 已注册进 `_EASYADAPTERS` + 一个预处理分支
      （training.py）+ inference.py 中泛化的检查点/输出/兜底。
- [ ] 在完整运行前，先用 `--limit 8` 的暂存批次肉眼做了 QA。

---

## 9. 进一步阅读

- **`easycontrol_adapters/colorization/README.md`**——完整的 colorize 设计笔记
  （标注策略、网点频段、Phase B 路线图）。上文所有内容的参考实现。
- **`docs/experimental/easycontrol.md`**——网络架构：双流前向、`b_cond` step-0 等价性基准、
  推理 KV 缓存、内存包络、局限性。改动 `network_args` 前请先读它。
- **`networks/methods/easycontrol.py`**——`EasyControlNetwork` + 打补丁后的
  `Block.forward` 闭包。对于一个新适配器，你*不应该*需要改动它；如果你觉得需要，请重新考虑那个
  差异是否真的存在于条件之中。
- **`networks/CLAUDE.md`**——逐模块映射图与分发不变量。
</content>
</invoke>
