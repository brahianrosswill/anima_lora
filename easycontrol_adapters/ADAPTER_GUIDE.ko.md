# 나만의 EasyControl 어댑터 만들기

Anima에 **새로운 EasyControl 컨트롤 태스크**를 직접 사용할 목적으로 추가하는 단계별
가이드입니다 — git 기여가 아니라, `easycontrol_adapters/<your_task>/` 아래에 존재하는
로컬 어댑터를 만드는 것입니다. 대표 예제인 **colorize**
(`easycontrol_adapters/colorization/`)를 상세히 따라가면서, 자신의 태스크를 위해
정확히 무엇을 바꿔야 하는지 짚어 줍니다.

단 하나만 읽는다면 이것입니다: EasyControl 어댑터는 **새로운 모델 코드가 아닙니다**.
네트워크(`networks/methods/easycontrol.py`), 2-스트림 포워드, `b_cond` logit-bias,
인퍼런스 KV 캐시 — 모두 이미 출하되어 공유됩니다. 하나의 컨트롤 태스크가 다른 모든
컨트롤 태스크와 다른 점은 *오직 한* 차원뿐입니다:

> **컨디션 이미지를 어떻게 만드는가.**

아래의 모든 내용은 이 단일 아이디어를 둘러싼 배관입니다.

---

## 0. 멘탈 모델 — EasyControl 어댑터란 실제로 무엇인가

EasyControl은 레퍼런스 이미지로 생성을 컨디셔닝합니다. 레퍼런스는 *cond 토큰*으로
VAE 인코딩되어 타깃 스트림과 나란히 모든 DiT 블록을 통과하며, 타깃 셀프-어텐션은
확장된 키 집합 `[target_k; cond_k]`에 어텐션합니다. (전체 아키텍처:
`docs/experimental/easycontrol.md`.)

**기본** EasyControl은 `cond == target`을 사용합니다 (레퍼런스가 곧 재구성 대상
이미지). **컨트롤 태스크 어댑터**는 이를 깹니다: 각 색상 타깃을 *다른* 컨디션
이미지와 짝지어, 모델이 항등(identity) 대신 `condition → target`을 학습하게 합니다.

| | 기본 EasyControl | 컨트롤 태스크 어댑터 (예: colorize) |
|---|---|---|
| target | image X | color image X |
| condition | image X (동일 레이턴트) | X의 *변환* (X의 흑백 망가) |
| 학습 내용 | 재구성 | manga → color |
| text 채널 | 전체 캡션 | (선택) 축소된 캡션 |

재사용 가능한 colorize의 통찰: **실제 흑백 망가에는 컬러 정답이 없으므로**,
`(흑백, 컬러)` 쌍을 수집해서 만들 수 없습니다. 방향을 *뒤집습니다* — 이미 보유한 컬러
이미지를 타깃으로 삼고, 각 이미지로부터 흑백 컨디션을 **합성**합니다(XDoG lineart +
알고리즘 기반 스크린톤). 컨디션을 인퍼런스 분포에 맞게 합성하는 것이 전부입니다.
가령 `depth → image`나 `pose → image`를 만든다면, "mangafy" 단계는 기존 학습 이미지에
대해 실행하는 depth estimator 또는 pose extractor가 됩니다.

따라서 할 일은: **함수 `color_image → condition_image`를 작성하고, 그 출력을 병렬
레이턴트 집합으로 캐시하고, 데이터셋 블루프린트가 이를 가리키게 하고, config + 한 줄짜리
selector 엔트리를 추가하는 것**입니다.

---

## 1. 건드리는 네 개의 표면(surface)

`<task>`라는 이름의 어댑터를 추가한다는 것은 정확히 다음을 만들거나 편집한다는
뜻입니다:

| # | 표면 | colorize 인스턴스 | 하는 일 |
|---|---------|-------------------|--------------|
| 1 | `easycontrol_adapters/<task>/` 프로젝트 | `colorization/` (`mangafy*.py`, `color_caption.py`, `prep.py`) | 컨디션을 만들고 캐시(선택적으로 축소된 텍스트 캐시도) |
| 2 | `configs/datasets/<task>.toml` | `configs/datasets/colorize.toml` | 타깃 레이턴트를 **cond_cache_dir**와 짝짓는 데이터셋 블루프린트 |
| 3 | `configs/methods/<task>.toml` (+ `configs/gui-methods/<task>.toml`) | `configs/methods/colorize.toml` | 메서드 config — 데이터셋을 가리키고 LR/epochs/`network_args` 설정 |
| 4 | `scripts/experimental_tasks/{training,inference}.py` selector | `_EASYADAPTERS = {"colorize"}` + 분기 | `EASYADAPTER=<task>`를 `make exp-easycontrol*` 타깃에 배선 |

순서대로 살펴봅니다. 여기 어느 것도 `networks/` 편집을 필요로 하지 않습니다.

---

## 2. 표면 1 — 어댑터 프로젝트 (`easycontrol_adapters/<task>/`)

실제 작업이 여기에 있습니다. 두 가지 일을 합니다: **컨디션 합성**과 **캐시**(추가로,
선택적으로 태스크별 텍스트 캐시 구축).

### 2a. 컨디션 합성기(synthesizer)

순수 함수: 컬러 RGB `uint8 (H,W,3)` + stem별 seed → **같은 크기**의 컨디션 RGB
`uint8 (H,W,3)`. (같은 크기가 중요합니다 — 토큰 카운트 매칭에 관한 §3 참조.) seed가
주어지면 결정적(deterministic)이어야 재실행과 병렬 워커가 비트 단위로 동일해집니다.

colorize에서 이는 `mangafy.py::mangafy_array`(및 그 CUDA 쌍둥이
`mangafy_gpu.py::mangafy_array_gpu`)입니다:

```python
# easycontrol_adapters/colorization/prep.py
Screener = Callable[[np.ndarray, int], np.ndarray]  # (img_rgb, seed) → cond_rgb
```

복제할 가치가 있는 핵심 설계 포인트:

- **stem별 결정적 seed.** colorize는 `zlib.crc32(stem)`을 사용합니다 — 프로세스마다
  salt가 붙어 워커들이 불일치하게 만드는 Python의 `hash()`가 *아닙니다*. seed jitter
  (colorize의 페이지별 톤 각도/주기)는 비결정성 없이 다양성을 줍니다.
- **무거운 import 지연.** colorize에는 세 개의 엔진(`cv2` / `gpu` / `sd`)이 있으며,
  실제로 어떤 페이지가 거기로 라우팅될 때만 3.5 GB짜리 SD 스택을 import합니다.
  합성기가 모델(depth net, line extractor)을 필요로 한다면, 다운로드 없는 폴백이 가볍게
  유지되도록 지연 import하세요.
- **모델 없는 폴백은 금쪽같습니다.** colorize의 `cv2`/`gpu` 엔진은 다운로드가
  전혀 필요 없습니다. 태스크에 이런 폴백이 있으면, `make exp-easycontrol-download`
  단계 없이 갓 체크아웃한 환경에서 prep + 학습이 가능합니다.
- **원자적 쓰기(atomic writes).** colorize의 `_save_png_atomic`은 임시 파일에 쓴 뒤
  `os.replace`를 하므로, 중단된 실행이 잘린 PNG를 남기지 않습니다 — 그렇지 않으면
  `out.exists()` 스킵 체크가 그 잘린 파일을 영원히 신뢰하게 됩니다. 이것을
  복사하세요; 안 그러면 실제로 부딪히는 버그입니다.

컨디션이 *실제* 아티팩트라면(이미 디스크에 depth map / sketch를 보유하고 있다면),
합성을 통째로 건너뛰고 encode 단계가 그것을 가리키게만 하면 됩니다. 합성은 타깃으로부터
컨디션을 도출해야 할 때만 필요합니다.

### 2b. (선택) 축소된 텍스트 캐시

colorize는 컨디션만 바꾸는 게 아니라 **캡션을 색상 태그 전용으로 축소**도 합니다
(`color_caption.py::filter_to_colors`). 그 근거는 태스크별이며 시사적입니다: 컨디션
(lineart + 스크린톤)이 이미 *공간적인 모든 것*을 인코딩하므로, 텍스트가 담을 유일한
것은 흑백이 줄 수 없는 단 하나 — **색조(hue)** 입니다. 캡션을 색상 태그로 필터링하면
살아남은 모든 토큰이 모델이 구조에서 얻을 수 없는 사실이 되어, 전체 캡션 속에 색상
태그가 파묻혀 약하게 스티어링되는 대신 강한 `prompt → color` 바인딩을 얻습니다.

자신의 태스크에 대해 같은 질문을 던지세요: **컨디션이 이미 결정하는 것은 무엇이고,
텍스트가 담아야 할 모호한 부분은 무엇인가?** `pose → image` 컨디션은 포즈는
고정하지만 외형/의상/배경은 고정하지 않으므로, 아마 *전체* 캡션을 유지할 것입니다.
`depth → image` 컨디션은 레이아웃은 고정하지만 정체성이나 팔레트는 고정하지 않습니다.
colorize의 "캡션을 잔여 모호성으로 필터링한다"는 동작은 *패턴*이지 요구사항이
아닙니다; 많은 어댑터는 캡션을 그대로 두고 텍스트 캐시를 통째로 건너뜁니다(데이터셋
블루프린트에서 `text_cache_dir`를 생략하기만 하면 됩니다 — §4).

캡션을 축소한다면, colorize의 두 개의 독립적인 노브에 유의하세요(혼동하지 마세요 —
README가 여기에 상당한 분량을 할애합니다):

- **`caption_dropout_rate`** — 자동 채색의 *바닥(floor)*. 약 5%의 스텝이 캡션을
  통째로 드롭하여(→ uncond), 빈 프롬프트 기본 동작을 학습시킵니다. **낮게**
  유지하세요(`0.05`); 비율이 높으면 무조건부(unconditional) 경로가 과하게 학습되어
  약한 스티어링으로 이어집니다.
- **`use_shuffled_caption_variants`** — 전체 대 부분의 *균형*. 텍스트 캐시는
  다중 변형이며(v0 = 전체 색상 집합, v1+ = 각 태그를 p≈0.5로 드롭한 셔플 버전),
  로더는 20% v0 / 80% v1+로 추출하므로 부분 프롬프트("pink hair" 단독)도
  동작합니다.

### 2c. `prep.py` — 캐시 빌더

세 개의 **멱등(idempotent)** 단계(이미 완료된 작업은 건너뜀; 재실행해도 안전):

1. **합성(Synthesize)** — `--src`(`post_image_dataset/resized`) 아래 모든 컬러
   이미지를 순회하며 합성기를 실행하고, 소스 하위 경로를 미러링하는 `--staging`
   디렉터리에 컨디션 PNG를 씁니다.
2. **인코드(Encode)** — 스테이징된 컨디션 이미지를 `library.preprocess.cache_latents`로
   각 이미지의 **네이티브 크기**에서 VAE 인코딩하여 `--cond_cache_dir`에 넣습니다.
   그래야 cond 레이턴트 shape가 타깃 레이턴트와 정확히 일치합니다. 타깃 캐시와 동일한
   `{stem}_{WxH}_anima.npz` 포맷입니다.
3. **(선택) 텍스트(Text)** — 캡션을 필터를 통해 다시 인코딩하여
   `--text_cache_dir`에 넣습니다. `library.preprocess.cache_text_embeddings`에
   `caption_transform=`(및 `caption_shuffle_variants` / `caption_tag_dropout_rate`)를
   주어 수행합니다.

라이브러리 프리미티브를 재사용하세요 — `library.preprocess.{cache_latents,
cache_text_embeddings, tqdm_progress}` 및 `library.preprocess._dataset.walk_images`.
encode 루프를 직접 짜지 마세요; `prep.py`는 `scripts/preprocess/*.py`와 똑같이,
이들 위에 얹힌 얇은 오케스트레이션 셸입니다.

colorize가 처리하는, 그 구조를 복사하면 공짜로 물려받는 두 가지 정확성 함정:

- **stem 키 매칭.** 텍스트 단계는 캡션 마스터(`image_dataset/`, `resized/`와 동일하게
  중첩됨)에서 `.txt`를 읽으므로, 결과 TE 캐시 경로가 로더의
  `image_dir=post_image_dataset/resized` 조회와 키 매칭됩니다. 캐시가 타깃 stem과
  키 매칭되지 않으면, 로더는 아무 말 없이 이들을 짝짓지 못합니다.
- **uncond 사이드카.** colorize의 텍스트 단계는 공유 `T5("")` uncond 사이드카를
  멱등하게 다시 스테이징합니다 — colorize 실행이 그것을 처음 건드리는 경우를
  대비해서입니다. 텍스트 캐시를 만들고 캡션 드롭아웃을 사용한다면 똑같이 하세요
  (`library.inference.uncond.stage_uncond_sidecar_with_models`).

---

## 3. 핵심 불변식 — cond 토큰 카운트는 타깃의 것과 일치해야 한다

DiT는 Anima의 **네이티브 shape 버킷팅**(두 개의 토큰 카운트 패밀리, 4032 / 4200;
CLAUDE.md 및 `docs/experimental/easycontrol.md` §"Cond token count" 참조) 위에서
동작합니다. **static-pad 노브가 없습니다** — cond 스트림은 cond 레이턴트의 네이티브
토큰 카운트로 실행됩니다.

이것이 §2a가 합성기 출력이 입력과 **같은 크기**임을 고집하고, §2c가 **네이티브
크기**로 인코딩하는 이유입니다: 그러면 컨디션 레이턴트가 자동으로 타깃 레이턴트와 같은
버킷 패밀리에 안착하고, `_extended_target_attention`이 그냥 동작합니다. 컨디션을
다운샘플한다면(정당함 — 더 작은 cond = 더 적은 메모리 / 더 빠름), 인코딩된 레이턴트가
여전히 실제 버킷에 안착하도록 이미지 레벨에서 *업스트림으로* 하세요; 네트워크에서
토큰 카운트를 제한하려 들지 마세요.

---

## 4. 표면 2 — 데이터셋 블루프린트 (`configs/datasets/<task>.toml`)

이것이 `cond ≠ target`을 만듭니다. **추가 서브셋 노브 하나**(`cond_cache_dir` 및
선택적으로 `text_cache_dir`)를 가진 평범한 데이터셋 블루프린트(`[general]` +
`[[datasets]]` + `[[datasets.subsets]]`)입니다.

colorize(`configs/datasets/colorize.toml`), 주석을 단 형태:

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

각 리다이렉트가 하는 일:

- **`cond_cache_dir`** — 컨트롤 태스크 어댑터를 구별짓는 유일한 노브. 로더는 여기서
  각 타깃을 컨디션 레이턴트와 stem 매칭합니다. 이것이 EasyControl의 2-스트림 포워드가
  레퍼런스로 소비하는 것입니다.
- **`text_cache_dir`** — **TE 전용** 리다이렉트입니다(레이턴트는 여전히 `cache_dir`에서
  옴). 전체 캡션을 유지한다면 통째로 생략하세요 — 그러면 로더가 공유 TE 캐시를 읽고
  `prep.py`의 텍스트 단계를 건너뜁니다.
- **`flip_aug = false`** — 필수. 뒤집힌 타깃은 뒤집힌 컨디션 레이턴트가 필요한데, 그건
  캐시하지 않았습니다. flip은 꺼 두세요.

colorize는 타깃 레이턴트와 TE에 **공유** `post_image_dataset/lora` 캐시를 재사용한다는
점에 유의하세요 — `make preprocess`가 이미 그것들을 만들었고, 아무것도 재인코딩하지
않습니다. 어댑터는 *컨디션* 캐시(그리고 어쩌면 축소된 텍스트 캐시)만 추가합니다.

---

## 5. 표면 3 — 메서드 config (`configs/methods/<task>.toml`)

`configs/methods/easycontrol.toml`의 거의 복제본입니다. 유일한 구조적 변경은
블루프린트를 가리키는 `dataset_config`이며, 나머지는 하이퍼파라미터입니다.

colorize(`configs/methods/colorize.toml`), 핵심이 되는 라인들:

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

자신의 태스크에 대해 고민할 노브들:

- **`b_cond_init`** — step-0 baseline-equivalence 초기값. `-10`은 cond가 step 0에서
  softmax 질량의 약 `e⁻¹⁰`만 기여하게 합니다(= baseline DiT, 이후 학습으로 올라감);
  `docs/experimental/easycontrol.md` §"Step-0 baseline equivalence"의 벤치가 이를
  도출합니다. colorize는 cond가 더 빨리 작동하도록 `-6`으로 완화합니다 — 강한 컨디션
  태스크는 그럴 여유가 있습니다. 어느 쪽이든 학습 가능합니다.
- **`easycontrol_cond_noise_max`** — cond 레이턴트에 대한 스텝별 학습 노이즈
  (σ ~ U(0, max), `cond + σ·ε`로 적용). `0` = cond가 완벽한 청사진; 양수 값은 cond를
  "손실성 힌트"로 저하시켜 텍스트가 잔여 디테일을 담도록 강제합니다. colorize는
  `0.02`를 사용합니다(아주 작음 — lineart *가 곧* 신호이므로). 기본 easycontrol.toml은
  `0.3`을 사용합니다.
- **`easycontrol_drop_p`** — image-CFG를 위한 배치별 full-cond 드롭아웃. colorize는
  `0`으로 설정합니다(컨디션은 항상 원함); 기본값은 `0.1`입니다.
- **`output_name`** — 고유한 stem이어야 합니다; 인퍼런스 selector가 이 이름으로 최신
  체크포인트를 해석합니다(§6).

선택적으로 `configs/gui-methods/<task>.toml`도 추가하세요 — `[variant]` 블록
(`family = "easycontrol"`, `label`, `description`, `order`)을 가진 자기 완결적
변형(토글 블록 없음)이며, 이를 추가하면 GUI의 EasyControl 탭 드롭다운에 나타납니다.
`configs/gui-methods/colorize.toml`을 참조하세요. CLI에서만 실행한다면 건너뛰세요.

---

## 6. 표면 4 — `EASYADAPTER=<task>`를 태스크 러너에 배선

`make exp-easycontrol*` 타깃은 `EASYADAPTER` 환경 변수로 디스패치합니다. 세 개의 작은
편집으로 `EASYADAPTER=<task>`가 자신의 config + prep + 체크포인트로 라우팅되게 합니다.

**`scripts/experimental_tasks/training.py`:**

1. 허용목록(allowlist)에 이름을 등록:
   ```python
   _EASYADAPTERS = {"colorize", "<task>"}   # was {"colorize"}
   ```
   (`_easyadapter()`가 이 집합에 대해 검증하고 오타에 에러를 냅니다.)

2. `cmd_easycontrol_preprocess`에서 전처리를 자신의 `prep.py`로 라우팅:
   ```python
   adapter = _easyadapter()
   if adapter == "colorize":
       run([PY, "easycontrol_adapters/colorization/prep.py", *extra]); return
   if adapter == "<task>":
       run([PY, "easycontrol_adapters/<task>/prep.py", *extra]); return
   ```

3. 학습 자체는 편집이 **필요 없습니다** — `cmd_easycontrol`이 이미
   `train(_easyadapter() or "easycontrol", extra)`를 하므로, 이름이 허용목록에 들어가면
   `EASYADAPTER=<task>`가 자동으로 `configs/methods/<task>.toml`을 실행합니다.

4. (합성기가 다운로드를 필요로 하는 경우에만) `cmd_easycontrol_download`에 자신의
   weight-fetch 태스크를 가리키는 분기를 추가하세요.

**`scripts/experimental_tasks/inference.py`** (`cmd_test_easycontrol`): selector가
현재 colorize를 하드코딩하고 있습니다. 자신의 태스크에 맞게 세 개의 colorize 전용
값 — 체크포인트 이름, 출력 하위 디렉터리, ref 폴백 디렉터리, 그리고 빈 프롬프트
기본값 — 을 일반화하세요:

```python
adapter = (os.environ.get("EASYADAPTER") or "").strip()
is_colorize = adapter == "colorize"
weight_name = "anima_colorize" if is_colorize else "anima_easycontrol"
out_sub     = "colorize"       if is_colorize else "easycontrol"
ref_fallback_dir = (ROOT/"post_image_dataset"/"resized") if is_colorize else (ROOT/"easycontrol-dataset")
```

자신의 `adapter == "<task>"` 케이스를 나란히 추가하세요(weight 이름은 `latest_output`
stem 매칭을 감안하여 config의 `output_name`과 일치해야 합니다). 자신의 태스크가
colorize처럼 빈 프롬프트 기본값과 특정 폴백 디렉터리의 실제 컨디셔닝 이미지를 원한다면,
아래쪽의 `is_colorize` 분기들을 그대로 따라 하세요.

---

## 7. 실행

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

전제 조건: `make preprocess`를 한 번 실행하여 공유 타깃 레이턴트 + TE가
`post_image_dataset/lora`에 존재해야 합니다(어댑터가 이를 재사용합니다).

### 알아 둘 만한 인퍼런스 설정 (colorize의 경험에서)

- **실제 in-distribution 컨디션을 입력하세요.** 레퍼런스는 인퍼런스 시 *있는
  그대로* VAE 인코딩됩니다 — 합성 없음. colorize는 실제 스크린톤이 입혀진 흑백
  페이지를 입력합니다; 밋밋한 그레이스케일 사진은 out-of-distribution이라 품질이
  저하됩니다. 합성기가 *흉내 낸* 것이 곧 인퍼런스가 받을 것으로 기대하는 것입니다.
- **`--easycontrol_image_match_size`** — 레퍼런스 종횡비에 맞는 토큰 버킷을 골라
  세로로 긴 페이지가 찌그러지지 않게 합니다. colorize는 이를 강제로 켭니다.
- **`--easycontrol_scale`** (`EC_SCALE=`, 구조 준수도) — 학습 기본값은 `1.0`;
  컨디션이 번지면 올리고(1.1–1.2), 더 느슨한 출력을 원하면 내리세요(0.7–0.8).
- **`--guidance_scale`** — 텍스트 정책과 상호작용합니다. colorize: 빈 프롬프트 → 낮은
  cfg(1.0–1.5, 밀어붙일 것이 없음); 텍스트 프롬프트 → 높은 cfg(3.0–4.5, 그것이
  프롬프트를 작동하게 만듦).

---

## 8. 체크리스트

- [ ] `easycontrol_adapters/<task>/`에 결정적이고 원자적으로 쓰는 합성기
      (`(img, seed) → cond` 동일 크기)와 멱등 `prep.py`
      (synthesize → encode → 선택적 text).
- [ ] 토큰 카운트가 타깃 버킷 패밀리와 일치하도록 컨디션을 **네이티브 크기**로
      인코딩(§3).
- [ ] `configs/datasets/<task>.toml`에 `cond_cache_dir`(+ 선택적
      `text_cache_dir`), `flip_aug = false`, 공유 타깃 캐시 재사용.
- [ ] `configs/methods/<task>.toml` → 블루프린트를 가리키고, 고유한
      `output_name`, `network_module = "networks.methods.easycontrol"`,
      `use_easycontrol = true`. (선택적 GUI 변형.)
- [ ] `EASYADAPTER=<task>`를 `_EASYADAPTERS`에 등록 + 전처리 분기
      (training.py) + inference.py의 일반화된 checkpoint/output/fallback.
- [ ] 전체 실행 전에 `--limit 8` 스테이징 배치를 눈으로 QA.

---

## 9. 더 읽을 곳

- **`easycontrol_adapters/colorization/README.md`** — colorize 설계 노트 전문
  (캡션 정책, 스크린톤 밴드, Phase B 로드맵). 위 모든 것의 레퍼런스 구현.
- **`docs/experimental/easycontrol.md`** — 네트워크 아키텍처: 2-스트림 포워드,
  `b_cond` step-0 equivalence 벤치, 인퍼런스 KV 캐시, 메모리 봉투, 한계. `network_args`를
  건드리기 전에 이것을 읽으세요.
- **`networks/methods/easycontrol.py`** — `EasyControlNetwork` + 패치된
  `Block.forward` 클로저. 새 어댑터를 위해 이것을 편집할 *필요는 없을* 것입니다;
  필요하다고 생각된다면, 그 차이가 정말로 컨디션에 있는 것인지 다시 생각해 보세요.
- **`networks/CLAUDE.md`** — 모듈별 맵과 디스패치 불변식.
