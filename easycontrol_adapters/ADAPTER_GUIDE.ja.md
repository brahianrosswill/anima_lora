# 自前の EasyControl アダプターを作る

Anima に **新しい EasyControl コントロールタスク** を追加するためのステップバイステップ
ガイドです。git へのコントリビューションとしてではなく、あくまで自分用に
`easycontrol_adapters/<your_task>/` の下に置くローカルアダプターとして追加します。
ここでは正準的な例である **colorize**
(`easycontrol_adapters/colorization/`) を詳しく辿り、自分のタスク向けに何を変えれば
よいのかを正確に示します。

一つだけ読むとすれば、これです: EasyControl アダプターは **新しいモデルコードではありません**。
ネットワーク (`networks/methods/easycontrol.py`)、two-stream forward、
`b_cond` logit-bias、推論時の KV キャッシュ — これらはすべて出荷済みで共有されています。
コントロールタスクが他のあらゆるコントロールタスクと異なるのは *たった一つ* の次元のみです:

> **コンディション画像をどう構築するか。**

以下のすべては、その単一のアイデアの周りの配管にすぎません。

---

## 0. メンタルモデル — EasyControl アダプターとは実際何なのか

EasyControl は参照画像に基づいて生成をコンディショニングします。参照は
VAE エンコードされて *cond トークン* となり、ターゲットストリームと並んで DiT の
あらゆるブロックを流れていきます; ターゲットのセルフアテンションは拡張された key 集合
`[target_k; cond_k]` にアテンションします。(完全なアーキテクチャは
`docs/experimental/easycontrol.md`。)

**デフォルト** の EasyControl は `cond == target` を使います (参照が再構成対象の画像
*そのもの*)。**コントロールタスクアダプター** はこれを破ります: 各色付きターゲットを
*異なる* コンディション画像とペアにし、モデルが恒等写像ではなく
`condition → target` を学習するようにします。

| | デフォルト EasyControl | コントロールタスクアダプター (例: colorize) |
|---|---|---|
| target | 画像 X | 色付き画像 X |
| condition | 画像 X (同一 latent) | X の *変換* (X の白黒マンガ) |
| 学習する内容 | 再構成 | マンガ → 色 |
| text チャネル | 完全キャプション | (任意) 削減したキャプション |

再利用できる colorize の洞察: **実際の白黒マンガには色の ground truth がありません**。
そのため `(白黒, 色)` ペアを収集して作ることはできません。そこで方向を *反転* します —
すでに手元にある色付き画像をターゲットとし、各画像から白黒コンディションを
**合成** するのです (XDoG lineart + アルゴリズム的スクリーントーン)。コンディションを
推論分布に合わせて合成することが、すべての肝です。たとえば `depth → image` や
`pose → image` を構築するなら、あなたの「mangafy」ステップは既存の学習画像に対して
実行する深度推定器やポーズ抽出器になります。

つまりあなたの仕事は: **関数 `color_image → condition_image` を書き、その出力を
並列の latent 集合としてキャッシュし、データセットブループリントをそこに向け、
config と一行のセレクターエントリを追加すること** です。

---

## 1. 触れる 4 つのサーフェス

`<task>` という名前のアダプターを追加するには、正確に以下を作成/編集します:

| # | サーフェス | colorize の実体 | 役割 |
|---|---------|-------------------|--------------|
| 1 | `easycontrol_adapters/<task>/` プロジェクト | `colorization/` (`mangafy*.py`, `color_caption.py`, `prep.py`) | コンディションを構築 + キャッシュする (任意で削減テキストキャッシュも) |
| 2 | `configs/datasets/<task>.toml` | `configs/datasets/colorize.toml` | ターゲット latent を **cond_cache_dir** とペアにするデータセットブループリント |
| 3 | `configs/methods/<task>.toml` (+ `configs/gui-methods/<task>.toml`) | `configs/methods/colorize.toml` | メソッド config — データセットを指し、LR/epochs/`network_args` を設定 |
| 4 | `scripts/experimental_tasks/{training,inference}.py` セレクター | `_EASYADAPTERS = {"colorize"}` + 分岐 | `EASYADAPTER=<task>` を `make exp-easycontrol*` ターゲットに配線する |

順番に見ていきます。ここでは `networks/` の編集は一切不要です。

---

## 2. サーフェス 1 — アダプタープロジェクト (`easycontrol_adapters/<task>/`)

ここが本当の作業の場所です。役割は二つ: **コンディションを合成する** ことと、
それを **キャッシュする** こと (加えて、任意でタスク固有のテキストキャッシュを構築する)。

### 2a. コンディションシンセサイザー

純粋関数です: 色付き RGB `uint8 (H,W,3)` + stem ごとの seed → **同じサイズ** の
コンディション RGB `uint8 (H,W,3)`。(同じサイズであることが重要です — トークン数の
マッチングについては §3 を参照。) seed が与えられれば決定論的でなければならず、
そうすることで再実行と並列ワーカーがビット単位で同一になります。

colorize ではこれが `mangafy.py::mangafy_array` (とその CUDA 版
`mangafy_gpu.py::mangafy_array_gpu`) です:

```python
# easycontrol_adapters/colorization/prep.py
Screener = Callable[[np.ndarray, int], np.ndarray]  # (img_rgb, seed) → cond_rgb
```

コピーする価値のある主要な設計ポイント:

- **stem ごとの決定論的 seed。** colorize は `zlib.crc32(stem)` を使います —
  Python の `hash()` *ではありません*。`hash()` はプロセスごとに salt されるため、
  ワーカー間で食い違いが生じます。seed ジッター (colorize ではページごとのトーン
  角度/周期) は非決定性なしにバリエーションを与えます。
- **重い import を遅延させる。** colorize は三つのエンジン (`cv2` / `gpu` / `sd`) を
  持ち、3.5 GB の SD スタックは実際にそこへルーティングされるページがある場合にのみ
  import します。シンセサイザーがモデル (深度ネット、線抽出器) を必要とするなら、
  遅延 import して no-download フォールバックを安価に保ちましょう。
- **no-model フォールバックは至宝です。** colorize の `cv2`/`gpu` エンジンは
  ダウンロードを一切必要としません。あなたのタスクにこれがあれば、`make
  exp-easycontrol-download` ステップなしでクリーンなチェックアウト上で prep +
  学習ができます。
- **アトミック書き込み。** colorize の `_save_png_atomic` は一時ファイルへ書き込んで
  `os.replace` するため、中断された実行が切り詰められた PNG を残すことが決してなく、
  `out.exists()` のスキップチェックがそれを永遠に信じてしまうことを防ぎます。これは
  コピーしてください; これをやらないと実際にぶつかるバグです。

コンディションが *実在の* アーティファクトである場合 (深度マップ/スケッチがすでに
ディスク上にある場合)、合成を完全にスキップしてエンコードステージをそれらに向けるだけで
済みます。合成が必要なのは、ターゲットからコンディションを導出しなければならない場合
だけです。

### 2b. (任意) 削減したテキストキャッシュ

colorize はコンディションを変えるだけでなく、**キャプションを色タグのみに削減** します
(`color_caption.py::filter_to_colors`)。その理由はタスク固有で示唆に富んでいます:
コンディション (lineart + スクリーントーン) はすでに *空間的なものすべて* をエンコード
しているため、テキストが運ぶべき唯一の残りものは白黒では得られない変数 — **色相** —
です。キャプションを色タグにフィルタすることで、生き残ったすべてのトークンが
構造からは得られない事実になり、完全キャプションの中に色タグが埋もれた弱いステアリング
ではなく、強い `prompt → color` 結合が得られます。

自分のタスクについても同じ問いを立ててください: **コンディションがすでに決定して
いるものは何で、テキストが運ぶべき曖昧さとして残るものは何か?** `pose → image`
コンディションはポーズを固定しますが、見た目/服装/設定は固定しません — だからおそらく
*完全* キャプションを保持するでしょう。`depth → image` コンディションはレイアウトを
固定しますが、アイデンティティやパレットは固定しません。colorize の「キャプションを
残余の曖昧さにフィルタする」という動きは *パターン* であって要件ではありません;
多くのアダプターはキャプションをそのまま保持し、テキストキャッシュを完全にスキップ
します (データセットブループリントから `text_cache_dir` を省くだけ — §4)。

キャプションを削減する場合、colorize の二つの独立したつまみに注意してください
(混同しないこと — README はここに実際の紙幅を割いています):

- **`caption_dropout_rate`** — 自動着色の *床*。約 5% のステップでキャプションを
  完全にドロップし (→ uncond)、空プロンプトのデフォルトを学習します。**低く** 保って
  ください (`0.05`); 高いレートは無条件パスを過学習させ、弱いステアリングにします。
- **`use_shuffled_caption_variants`** — 完全 vs 部分の *バランス*。テキストキャッシュは
  マルチバリアント (v0 = 完全な色集合、v1+ = 各タグを p≈0.5 でドロップしたシャッフル版)
  で、ローダーは v0 を 20% / v1+ を 80% 引くため、部分プロンプト (「pink hair」単独)
  が機能します。

### 2c. `prep.py` — キャッシュビルダー

三つの **冪等** ステージ (完了済みの作業はスキップ; 再実行しても安全):

1. **合成** — `--src` (`post_image_dataset/resized`) 以下のすべての色付き画像を
   走査し、シンセサイザーを実行し、コンディション PNG をソースのサブパスを鏡写しに
   した `--staging` ディレクトリへ書き込みます。
2. **エンコード** — ステージングされたコンディション画像を
   `library.preprocess.cache_latents` 経由で `--cond_cache_dir` へ VAE エンコード
   します。各画像の **ネイティブサイズ** で行うことで、cond latent の形状がその
   ターゲット latent と正確に一致します。ターゲットキャッシュと同じ
   `{stem}_{WxH}_anima.npz` フォーマットです。
3. **(任意) テキスト** — キャプションをあなたのフィルタを通して再エンコードし、
   `library.preprocess.cache_text_embeddings` 経由で `--text_cache_dir` へ書き込み
   ます (`caption_transform=` と `caption_shuffle_variants` /
   `caption_tag_dropout_rate` を指定)。

ライブラリのプリミティブを再利用してください — `library.preprocess.{cache_latents,
cache_text_embeddings, tqdm_progress}` と `library.preprocess._dataset.walk_images`。
エンコードループを手で組まないこと; `prep.py` はそれらの上の薄いオーケストレーション
シェルであり、`scripts/preprocess/*.py` とまったく同じです。

colorize が処理している二つの正しさの落とし穴。その構造をコピーすれば無償で
継承できます:

- **stem キーマッチング。** テキストステージはキャプションマスター
  (`image_dataset/`、`resized/` と同一にネストされている) から `.txt` を読むため、
  結果として得られる TE キャッシュパスがローダーの
  `image_dir=post_image_dataset/resized` ルックアップとキーマッチします。キャッシュが
  ターゲット stem とキーマッチしない場合、ローダーは黙ってそれらをペアにしません。
- **uncond サイドカー。** colorize のテキストステージは共有された `T5("")` uncond
  サイドカーを冪等に再ステージします。colorize 実行がそれに最初に触れる場合に備えて
  です。テキストキャッシュを構築してキャプションドロップアウトを使うなら、同じことを
  してください (`library.inference.uncond.stage_uncond_sidecar_with_models`)。

---

## 3. 重要な不変条件 — cond トークン数はターゲットのものと一致しなければならない

DiT は Anima の **ネイティブ形状バケッティング** で動作します (二つのトークン数
ファミリー、4032 / 4200; CLAUDE.md と `docs/experimental/easycontrol.md` の
"Cond token count" を参照)。**static-pad のつまみは存在しません** — cond ストリームは
cond latent のネイティブトークン数で動きます。

これこそが、§2a がシンセサイザー出力を入力と **同じサイズ** にするよう主張し、
§2c が **ネイティブサイズ** でエンコードする理由です: そうすればコンディション
latent は自動的にそのターゲット latent と同じバケットファミリーに着地し、
`_extended_target_attention` がそのまま動きます。コンディションをダウンサンプルする
場合 (正当です — 小さい cond = より少ないメモリ / より高速)、画像レベルで *上流* に
行ってください。そうすればエンコードされた latent はやはり実在のバケットに着地します;
ネットワーク内でトークン数を抑え込もうとしないこと。

---

## 4. サーフェス 2 — データセットブループリント (`configs/datasets/<task>.toml`)

これが `cond ≠ target` を成立させるものです。通常のデータセットブループリント
(`[general]` + `[[datasets]]` + `[[datasets.subsets]]`) に **一つの追加の subset
つまみ** が付いたものです: `cond_cache_dir` (と任意で `text_cache_dir`)。

colorize (`configs/datasets/colorize.toml`)、注釈付き:

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

各リダイレクトの役割:

- **`cond_cache_dir`** — コントロールタスクアダプターを区別する唯一のつまみ。
  ローダーは各ターゲットをここのコンディション latent に stem マッチします。これが
  EasyControl の two-stream forward が参照として消費するものです。
- **`text_cache_dir`** — **TE のみ** のリダイレクト (latent は依然として
  `cache_dir` から来ます)。完全キャプションを保持するなら完全に省いてください —
  するとローダーは共有 TE キャッシュを読み、`prep.py` のテキストステージをスキップ
  できます。
- **`flip_aug = false`** — 必須。反転されたターゲットには反転されたコンディション
  latent が必要ですが、それはキャッシュしていません。flip はオフのままにして
  ください。

colorize はターゲット latent と TE に **共有** の `post_image_dataset/lora` キャッシュを
再利用していることに注意してください — `make preprocess` がすでにそれらを構築済みで、
何も再エンコードされません。あなたのアダプターは *コンディション* キャッシュ (と
たぶん削減テキストキャッシュ) を追加するだけです。

---

## 5. サーフェス 3 — メソッド config (`configs/methods/<task>.toml`)

`configs/methods/easycontrol.toml` のほぼクローンです。構造上の唯一の変更は
`dataset_config` をあなたのブループリントに向けることだけ; 残りはハイパーパラメータ
です。

colorize (`configs/methods/colorize.toml`)、load-bearing な行:

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

自分のタスク向けに考えるべきつまみ:

- **`b_cond_init`** — step-0 のベースライン等価初期化。`-10` は step 0 で cond が
  softmax 質量の約 `e⁻¹⁰` だけ寄与するようにします (= ベースライン DiT、そこから学習
  して上げていく); `docs/experimental/easycontrol.md` の "Step-0 baseline
  equivalence" のベンチがこれを導出しています。colorize はこれを `-6` に緩め、cond が
  より早く効くようにします — より強いコンディションのタスクならそれを許容できます。
  いずれにせよ学習可能です。
- **`easycontrol_cond_noise_max`** — cond latent に対するステップごとの学習時ノイズ
  (σ ~ U(0, max)、`cond + σ·ε` として適用)。`0` = cond は完璧なブループリント;
  正の値はそれを「ロッシーなヒント」に劣化させ、テキストに残余ディテールを運ばせます。
  colorize は `0.02` を使います (極小 — lineart *こそ* が信号です)。デフォルトの
  easycontrol.toml は `0.3` を使います。
- **`easycontrol_drop_p`** — image-CFG 用のバッチごとの完全 cond ドロップアウト。
  colorize は `0` に設定します (コンディションは常に欲しい); デフォルトは `0.1`。
- **`output_name`** — 一意な stem でなければなりません; 推論セレクターはこの名前で
  最新のチェックポイントを解決します (§6)。

任意で `configs/gui-methods/<task>.toml` も追加できます — `[variant]` ブロック
(`family = "easycontrol"`、`label`、`description`、`order`) を持つ自己完結型の
バリアント (トグルブロックなし) で、GUI の EasyControl タブのドロップダウンに表示
されるようになります。`configs/gui-methods/colorize.toml` を参照してください。
CLI からのみ実行するなら、これはスキップしてください。

---

## 6. サーフェス 4 — `EASYADAPTER=<task>` をタスクランナーに配線する

`make exp-easycontrol*` ターゲットは `EASYADAPTER` 環境変数でディスパッチします。
三つの小さな編集で `EASYADAPTER=<task>` をあなたの config + prep + チェックポイントへ
ルーティングします。

**`scripts/experimental_tasks/training.py`:**

1. 名前を allowlist に登録します:
   ```python
   _EASYADAPTERS = {"colorize", "<task>"}   # was {"colorize"}
   ```
   (`_easyadapter()` はこの集合に対して検証し、タイポでエラーになります。)

2. `cmd_easycontrol_preprocess` で preprocess をあなたの `prep.py` にルーティング
   します:
   ```python
   adapter = _easyadapter()
   if adapter == "colorize":
       run([PY, "easycontrol_adapters/colorization/prep.py", *extra]); return
   if adapter == "<task>":
       run([PY, "easycontrol_adapters/<task>/prep.py", *extra]); return
   ```

3. 学習自体には **編集不要** です — `cmd_easycontrol` はすでに
   `train(_easyadapter() or "easycontrol", extra)` をしているため、名前が allowlist に
   入りさえすれば `EASYADAPTER=<task>` は自動的に `configs/methods/<task>.toml` を
   実行します。

4. (シンセサイザーがダウンロードを必要とする場合のみ) `cmd_easycontrol_download` に
   あなたの重みフェッチタスクを指す分岐を追加します。

**`scripts/experimental_tasks/inference.py`** (`cmd_test_easycontrol`): セレクターは
現在 colorize をハードコードしています。あなたのタスク向けに三つの colorize 固有の
値を一般化してください — チェックポイント名、出力サブディレクトリ、ref フォールバック
ディレクトリ、そして空プロンプトのデフォルト:

```python
adapter = (os.environ.get("EASYADAPTER") or "").strip()
is_colorize = adapter == "colorize"
weight_name = "anima_colorize" if is_colorize else "anima_easycontrol"
out_sub     = "colorize"       if is_colorize else "easycontrol"
ref_fallback_dir = (ROOT/"post_image_dataset"/"resized") if is_colorize else (ROOT/"easycontrol-dataset")
```

あなたの `adapter == "<task>"` ケースを隣に追加してください (weight name は
`latest_output` の stem マッチを除いて、config の `output_name` と一致しなければ
なりません)。あなたのタスクが colorize のように空プロンプトのデフォルトと特定の
フォールバックディレクトリからの実在のコンディショニング画像を欲するなら、もっと下の
`is_colorize` 分岐を真似てください。

---

## 7. 実行する

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

前提条件: `make preprocess` を一度実行し、共有ターゲット latent + TE が
`post_image_dataset/lora` に存在するようにすること (あなたのアダプターはそれらを
再利用します)。

### 知っておく価値のある推論設定 (colorize の経験から)

- **実在の in-distribution なコンディションを与える。** 参照は推論時に *そのまま* VAE
  エンコードされます — 合成はされません。colorize は実在のスクリーントーン付き白黒
  ページを与えます; のっぺりしたグレースケール写真は out-of-distribution であり劣化
  します。あなたのシンセサイザーが *模倣した* ものこそ、推論が受け取ることを期待する
  ものです。
- **`--easycontrol_image_match_size`** — 参照のアスペクト比に一致するトークンバケットを
  選び、縦長ページが潰れないようにします。colorize はこれを強制オンにします。
- **`--easycontrol_scale`** (`EC_SCALE=`、構造への追従) — `1.0` が学習時のデフォルト;
  コンディションが滲むなら上げ (1.1–1.2)、より緩い出力には下げます (0.7–0.8)。
- **`--guidance_scale`** — あなたのテキストポリシーと相互作用します。colorize: 空
  プロンプト → 低 cfg (1.0–1.5、押す先がない); テキストプロンプト → 高め
  (3.0–4.5、プロンプトを効かせるのがこれ)。

---

## 8. チェックリスト

- [ ] 決定論的でアトミック書き込みのシンセサイザー (`(img, seed) → cond` 同一サイズ)
      と冪等な `prep.py` (合成 → エンコード → 任意のテキスト) を持つ
      `easycontrol_adapters/<task>/`。
- [ ] コンディションを **ネイティブサイズ** でエンコードし、トークン数がターゲット
      バケットファミリーと一致すること (§3)。
- [ ] `cond_cache_dir` (+ 任意で `text_cache_dir`)、`flip_aug = false` を持ち、共有
      ターゲットキャッシュを再利用する `configs/datasets/<task>.toml`。
- [ ] ブループリントを指し、一意な `output_name`、`network_module =
      "networks.methods.easycontrol"`、`use_easycontrol = true` を持つ
      `configs/methods/<task>.toml`。(任意で GUI バリアント。)
- [ ] `_EASYADAPTERS` に登録された `EASYADAPTER=<task>` + preprocess 分岐
      (training.py) + inference.py での一般化されたチェックポイント/出力/フォール
      バック。
- [ ] フル実行の前に `--limit 8` のステージングバッチを目視で QA 済み。

---

## 9. さらに読むには

- **`easycontrol_adapters/colorization/README.md`** — colorize の設計ノート全文
  (キャプションポリシー、スクリーントーンの帯、Phase B ロードマップ)。上記すべての
  リファレンス実装。
- **`docs/experimental/easycontrol.md`** — ネットワークアーキテクチャ: two-stream
  forward、`b_cond` step-0 等価性ベンチ、推論時 KV キャッシュ、メモリエンベロープ、
  制限事項。`network_args` に触れる前にこれを読んでください。
- **`networks/methods/easycontrol.py`** — `EasyControlNetwork` + パッチされた
  `Block.forward` クロージャ。新しいアダプターのためにこれを編集する必要は *ない*
  はずです; 必要だと思ったら、その違いが本当にコンディションに宿っているのかを再考
  してください。
- **`networks/CLAUDE.md`** — モジュール別マップとディスパッチ不変条件。
