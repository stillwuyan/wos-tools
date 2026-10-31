# 自动化配置说明

## 状态与路径

`tasks/common.yaml` 和 `tasks/intel.yaml` 是**无真实素材、未经实机验证的示例骨架，不能直接运行**。场景、按钮、出口、分支、尺寸、固定坐标及超时均为占位，必须采集并逐项验证；示例不证明情报已领取，也不是完整业务流程。默认任务结果为 `unverified`。

marker 模板固定保存在 `captures/templates/`，任务示例图片使用 `tasks/templates/`。采集后需手动整理任务所需图片并调整 YAML。

路径约定中的 home 是项目根目录，即 `app.py` 所在目录，与从哪里启动无关。`run` 的任务列表路径、列表 YAML 的 `common` 和 `tasks[].file`，以及 common 和任务 YAML 中所有图片引用（包括场景 `image`、`click_image`、`expect.image` 和 `when.image`）的相对路径均以项目根目录为基准，而不是以引用它们的 YAML 所在目录为基准。例如 `tasks/common.yaml` 和 `tasks/templates/town.png` 分别指项目根目录下的文件。

`--config`、`--server-jar`、配置中的 `scrcpy.server_jar`、环境变量 `SCRCPY_SERVER_PATH` 及 `mark` 的 `--output-dir` 的相对路径也以项目根目录为基准；配置文件的位置不会改变 jar 路径基准。程序不会将 `~` 展开为用户目录，`~/file` 会作为项目根目录下的相对路径处理。绝对路径直接指向其指定位置，不再拼接项目根目录；Windows 路径须使用完整绝对路径，例如 `C:\data\file.yaml`，不能使用 `C:file.yaml` 或 `\data\file.yaml`。

`run` 可选 `--trace-scenes`：每次场景识别记录全场景模板得分（`captures/scene-trace/scene-trace.jsonl`，每帧一条），命中集合变化或发生 `SceneConflict` 时落盘截图 `frame_*.png` 及同名 sidecar JSON，运行结束输出 covers 审计（每条 `A covers B` 声明在多少真实帧上被实际行使，未行使的声明会标注 covered 场景单独命中的帧数，用于核对 covers 配置是否真实反映界面遮挡关系）。追踪只读，不改变识别与执行语义；`SceneConflict` 仍按原样抛出停止。

`run` 仅在最终生效的 `screen_on: false` 时显示只读预览窗口；关闭窗口不会停止任务。设为 `true` 时不创建预览窗口，但视频流仍用于自动化识别。`mark` 的采集窗口始终显示，不受该规则限制。在终端按 `Ctrl+C` 可请求停止任务并清理连接。任务等待会被唤醒，但已开始的设备操作须先结束，强制中断不保证设备状态恢复。

安装 `requirements.txt`，准备 ADB、已授权设备和 scrcpy 3.3.4 server，在 `config.yaml` 设置 `scrcpy.server_jar` 或指定 `--server-jar`。可选在 `config.yaml` 设置 `scrcpy.tap_jitter_pixels`（非负整数，单位设备像素）：所有 `click`/`click_image`（含 `cmd` 单发命令）的实际点击点在该值内随机偏移并钳制到屏幕边界，0 或省略为精确坐标；`swipe` 与 `back` 不受影响。

`scrcpy.screen_on` 默认 `true`，不会发送关屏或亮屏请求，也不保证屏幕常亮。设为 `false` 或使用 `--screen-off` 时，连接后通过控制通道请求关闭物理屏幕，正常停止时请求亮屏。消息没有设备执行确认，能否生效取决于设备；异常中断或断连时不保证恢复。物理关屏与系统休眠不同，视频流通常可继续供 `mark` 采集和 `run` 的只读预览窗口使用。

为保留设备端 jar，server 使用 `cleanup=false`。`scrcpy.stay_awake` 默认 `false`；即使设为 `true` 或使用 `--stay-awake`，scrcpy 3.3.4 也不会实际设置保持唤醒。设备可能休眠、视频流可能中断，程序不会修改或恢复设备的休眠设置。全局配置和命令行选项均适用于 `mark`、`run`、`cmd`。采集并完成配置及实机验收后，才可从仓库根目录执行：

```shell
python app.py run tasks/auto_polar_terror.yaml
```

从其他工作目录也可使用脚本的绝对路径调用，例如在 PowerShell 中（将路径替换为实际项目路径）：

```powershell
python "C:\path\to\wos-tools\app.py" run tasks/auto_polar_terror.yaml --config config.yaml
python "C:\path\to\wos-tools\app.py" mark --output-dir captures
```

这些相对参数及 YAML 内的相对引用仍指向项目根目录下的资源，而不是调用命令的目录。

`run` 先加载并校验全部 YAML 和图片，再连接设备；缺图时在连接前失败。加载器拒绝未知字段、重复键及无效引用，也会解码图片：模板必须比帧小且灰度有纹理。连接后还会核对帧和设备尺寸，静态校验不能证明素材或业务流程正确。

## 采集与单次命令

```shell
python app.py mark --serial DEVICE_SERIAL --output-dir captures
python app.py cmd --serial DEVICE_SERIAL back
python app.py cmd --serial DEVICE_SERIAL click 100 200
python app.py cmd --serial DEVICE_SERIAL swipe 100 500 100 200 --duration 500
```

`mark` 左键点选采集点、拖框采集模板，右键或 `s` 保存，`Ctrl+Z` 撤销本次会话的上一次保存，`q` 或 `Esc` 退出，`h` 切换帮助。每次启动生成带唯一后缀的 `mark-records_*.yaml`，路径见日志；`--output-dir` 只改变记录文件输出目录，默认是项目根目录下的 `captures/`，也可指定绝对目录。模板始终生成在项目根目录下的 `captures/templates/`，不随 `--output-dir` 改变，记录的 `image` 是相对项目根目录的路径。撤销模板记录会删除对应图片。

记录仅供采集和人工抄录，任务示例不读取 marker 记录文件或 `step_001` 资源编号。首次保存确定 `meta.frame_size` 和 `meta.device_size`，各含 `width`、`height`；手动抄为 common 的 `[width, height]`，同一会话之后尺寸变化会拒绝保存，换尺寸需新会话。用户须手动将采集模板从 `captures/templates/` 复制到 `tasks/templates/`，并调整 common 和任务 YAML 的图片引用。点的 `device.x`、`device.y` 填为 `[x, y]`，滑动两端同理，不能使用窗口或帧坐标。示例图片名和固定坐标均须替换；若整理或重命名图片，配置也需同步更新。方向、分辨率、缩放或 UI 变化后重新采集并验收。

`cmd` 每次连接设备发送一次命令，不识别场景或确认结果；点击/滑动采用 device 坐标并检查边界，滑动 `--duration` 为正毫秒，默认 300。点击同样应用 `scrcpy.tap_jitter_pixels` 随机偏移（如已配置）。连接选项放在 `back`、`click`、`swipe` 之前。`mark`、`run`、`cmd` 都支持 `--config`、`--server-jar`、`--adb`、`--serial`、`--host`、`--port`、`--socket-name`；不要与同设备其他 scrcpy 会话并发，自定义 socket 名称不保证与服务端匹配。发送成功不等于设备已处理。

## 配置字段

列表根节点为 `common` 加上 `tasks` 或 `loop` 二者之一（必须且只能有一个），可选 `scrcpy` 配置整个运行会话：

```yaml
common: tasks/common.yaml
scrcpy:
  screen_on: true
  stay_awake: false
tasks:
  - file: tasks/polar_terror.yaml
    on_failure: stop
```

两个字段均为布尔值，可单独省略。优先级为命令行 > 任务列表 `scrcpy` > 全局 `config.yaml` > 默认值（`screen_on: true`，`stay_awake: false`）。`--screen-off` 覆盖为 `screen_on: false`，`--stay-awake` 覆盖为 `stay_awake: true`；未提供选项时沿用 YAML。若要撤销 YAML 中相反的设置，需修改相应 YAML。选项作用于整个任务列表或循环，不支持每个子任务单独切换；`mark` 和 `cmd` 不读取任务列表配置。旧配置 `turn_screen_off` 应替换为取反后的 `screen_on`（例如 `turn_screen_off: true` 改为 `screen_on: false`）。

任务执行结束产生一个结果：`verified`（有独立可靠的业务成功证据）、`unverified`（流程走完但无独立证据，需人工核查，默认）、`failed`（确认失败）、`uncertain`（无法判断，保守兜底）、`suspended`（计划内挂起，如无可执行队列，由上层再试）。结果值与 `reason`（英文短句说明原因）一起由 `loop` 的 `success`/`retry` 精确匹配。

`tasks` 按顺序执行，每项为 `file` 和 `on_failure: stop|skip`。`verified`、`unverified` 和 `suspended` 都继续；只有 `failed` 可按 `skip` 跳过，`uncertain` 或异常停止。跳过且后面还有任务时先尝试收尾；列表正常结束也会收尾，停止或异常不保证恢复。

`loop` 把单个任务重复执行到成功次数达标，字段：

- `count: 正整数` 需要成功完成的轮数；`file` 循环执行的任务文件；`on_failure: stop|skip`（默认 `stop`）处理意外结果：`uncertain` 总是停止，`failed` 在 `skip` 时跳过该轮不计成功继续，`stop` 时停止。
- `success: {result: verified|unverified|suspended, reason: 字符串}` 定义"成功一次"的判定，`result` 和 `reason`（省略为空）须同时匹配任务结果；每成功一次计入 `count`。
- `wait: 非负秒数`（默认 0）每成功一次后等待的秒数，最后一次成功后不等待。
- `retry: [{result: unverified|suspended, wait: 非负秒数}]` 按任务结果 `result` 匹配的挂起分支（不看 `reason`）：结果匹配时等待 `wait` 秒后重跑，**不计入** `count`；`result` 不得重复。
- `max_retries: 非负整数`（默认 0，配置 `retry` 时必须为正）两次成功之间连续重试上限，达到后停止；每成功一次计数清零。`retry` 与 `max_retries` 用于队列等可恢复的暂时性退出。

循环在每轮开始前像 `tasks` 一样 prepare（必要时切主场景），全部成功达到 `count` 后统一收尾一次；停止或异常不保证恢复。成功判定依赖任务结果的 `result` + `reason` 文本，任务分支结果变更时须同步更新 `loop.success` / `loop.retry`。示例 `tasks/auto_polar_terror.yaml`：巨兽任务循环，集结发出（`verified`）计一次、成功间隔 3 分钟、无队列（`suspended`）等 1 分钟重试且不计次、连续 10 次无队列停止。长间隔执行期间设备若熄屏，下一轮无法识别画面会干净停止，需亮屏后重跑。

`common.yaml` 根节点为 `device`、`scenes`、`switches`：

- `device.frame_size`、`device.device_size` 是两个正整数的 `[width, height]`，来自 marker 的 `meta`。
- `scenes` 是名称到场景的映射，必须有 `town`、`wild` 且均为 `type: main`。其他场景可为 `main` 或 `sub`；每项必填 `type` 和模板 `image`，可选 `threshold`（0 到 1，默认 0.8）及帧坐标 `region: [x, y, width, height]`（在帧内且不小于模板，省略则全帧）。
- 只有 `sub` 可写 `covers`，表示遮盖的场景，关系可传递但不能成环。同帧匹配后移除被覆盖场景，剩余多个会报冲突，不按顺序选择；覆盖应反映真实界面，不能用来掩盖误识别。
- 可选 `exit` 是动作列表；`switches` 每项为 `from`、`to`、`steps`，仅允许不同主场景间唯一有向对。出口和切换动作支持 `click`、`click_image`、`swipe`、`back`、`wait`、`wait_scene`。每两个副作用之间必须有 `wait_scene`，至少一个副作用且末尾必须是 `wait_scene`。出口末尾只能等待主场景，切换末尾只能等待唯一目标场景。`wait_scene: {scenes: [场景名], timeout: 正秒数}`，场景动作可等待多个场景；未知场景可在超时内继续等待，冲突或超时会停止。示例出口及切换方向只是待验证的假设。

任务根节点为 `task`（名称）、`start`（主场景名或非空主场景列表）、`steps`（非空顺序步骤列表），可选 `result: verified|unverified`，省略为 `unverified`。根级 `verified` 要求最后一步为带 `expect` 的操作或 `wait_scene`；这只能保证配置了画面条件，不能证明该条件是可靠的业务成功证据。启动前会尝试退出子场景并在必要时选择唯一切换到允许的主场景。步骤如下：

- `click: [x, y]` 固定点击 device 点；`click_image: tasks/templates/button.png` 在全帧按模板匹配并点击匹配中心；`swipe: {from: [x, y], to: [x, y], duration: 正秒数}`；`back: true`。这些副作用步骤都必须带 `require_scene: 场景名`，执行前重新检查场景。固定坐标不会识别目标，必须实测调整。
- `wait: 非负秒数` 仅等待并获取健康帧，不确认业务结果，不能附带场景或期望字段。
- 任务步骤 `wait_scene: {scenes: [场景名], timeout: 正秒数}` 恰好只能等待一个场景，超时为 `uncertain`。
- 副作用步骤可带 `expect: {scene: 名称}` 或 `expect: {image: 路径}` 等待结果；可选 `timeout`（正秒数，默认 10）、`on_timeout: failed|uncertain`（默认 `uncertain`）和 `retry: {count: 整数, interval: 正秒数}`。没有 `expect` 时不能写这些选项，动作后直接进入下一步，并不证明动作成功。
- `retry.count` 为额外重试机会上限 1 到 100，每轮有独立的 `timeout`，并非总时限。超时后等待间隔，在新帧重新检查期望和原 `require_scene`，发送前再次检查；不保证每次机会都会发送。超时不代表设备未收到命令，领取、购买等不可安全重放的动作不要重试。
- `detect: {branches: [...], timeout: 正秒数, on_timeout: failed|uncertain}` 等待分支。每个分支必填 `when: {scene: 名称}` 或 `when: {image: 路径}`，并且只能选择 `result: verified|unverified|failed|uncertain|suspended`（可选字符串 `reason`）或非空 `steps`。结果立即结束任务；分支 `steps` 执行完后接回该 `detect` 后的后续步骤，最后才使用任务默认 `result`。同一 detect 中条件不得重复，须采集互斥证据。

图片条件需要当前有已识别场景，但不限定具体场景；场景未知时等待，不检查图片，截止仍未知返回 `uncertain`。同帧多个分支命中返回 `uncertain`，不按顺序选取。图片条件和 `click_image` 使用全帧、默认阈值 0.8；匹配点击会拒绝缺失或多个分离峰，但邻近目标仍可能被当作一个，不能保证物理目标唯一。场景冲突、尺寸变化、断流、目标缺失或控制异常会停止；结果仅在截止后观察到、截止时没有新帧等也可能为 `uncertain`。任务 `wait_scene` 与场景出口/切换中的等待超时行为不同，勿混淆。

## 安全验收

示例假定从 wild 打开 intel，空情报直接 `unverified`；检测到领取按钮才点击，等待弹窗后用占位 device 坐标关闭，再等待 intel 并以默认 `unverified` 结束。没有真实素材或独立成功证据，不能将这视为已领取；空提示与按钮必须互斥，检测和点击之间画面仍可能变化。仅有独立可靠的业务证据才可使用 `verified`。

先人工确认账户和资源，避免付费、消耗及不可逆操作；逐一核对模板、场景覆盖、出口、切换、固定点、分支和超时，再低风险单步验收，包括遮挡、延迟、断流及中断。控制后新帧也不保证游戏状态已更新；不要通过放宽阈值或盲点绕过未知场景。`Ctrl+C` 返回 130 并尝试关闭会话，但不保证回到主场景；异常、超时或中断后设备可能已执行部分动作，必须人工检查。正常完成且收尾成功退出码为 0，否则为 1；`unverified` 仍需人工核查。
