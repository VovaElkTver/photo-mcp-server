# photo-mcp-server

Асинхронный **MCP-сервер** (Model Context Protocol) для фотофиксационного анализа
1D/2D-кодов (QR, DataMatrix, штрихкоды). Сервер отдаёт три инструмента LLM-хосту
по транспорту **SSE** (сеть, `0.0.0.0:8000`), а локальная модель (vLLM) решает,
какие инструменты вызвать, и получает реальные результаты — не догадки.
Транспорт SSE (вместо stdio) нужен для удалённого выполнения: vLLM может
подключиться к серверу через Responses API (`--tool-server`).

```
+------------------+         SSE (HTTP / JSON-RPC)        +------------------+
|  LLM host (vLLM) |  <---- tools/call (/messages/) ->  | photo_mcp_server |
|  Inferact/Qwen3.8|  <---- results (/sse) ---------  |  | (FastMCP :8000)|
|  -27B-NVFP4 :8001|                                   +------------------+
 +------------------+                                        |
                                                            | detect_code (YOLO)
                                                            | align_perspective (cv2)
                                                            | decode_code (cascade)
```

---

## 1. Состав

| Файл | Назначение |
|------|-----------|
| `photo_mcp_server.py` | MCP-сервер (FastMCP, SSE, `0.0.0.0:8000`). 3 инструмента. |
| `test_mcp_model.py`   | Тест-скрипт: MCP-агент, где модель на 8001 вызывает инструменты на картинках из `test-imgs/`. |
| `test-imgs/`          | Исходные фото с кодами (тестовые образцы). |
| `wechat_models/`      | Авто-скачанные модели WeChatQRCode (создаются при первом вызове). |
| `yolov8n-barcode.pt`  | Модель YOLO (создаётся при первом вызове `detect_code`). |
| `mcp_server.log`      | Лог сервера (весь лог идёт сюда + stderr, **никогда не в stdout**). |

---

## 2. Зависимости

Установите в отдельное окружение (проверенные версии):

```bash
pip install "mcp==1.29.0"            # FastMCP: ОБЯЗАТЕЛЬНО mcp<2 (в mcp 2.0 убрано mcp.server.fastmcp)
pip install "opencv-contrib-python==4.14.0.94"   # cv2.wechat_qrcode_* (не 5.0 — там нет InferenceEngine)
pip install "ultralytics==8.4.121"
pip install "torch"                  # CUDA-сборка (cu130) при наличии GPU
pip install "pylibdmtx==0.1.9"       # не 0.1.10 — там пустой модуль (нет decode)
pip install "zxing-cpp"
pip install "pyzbar"
```

> **Важно про версии:**
> - `mcp` должно быть **< 2** (в 1.29.0 есть `mcp.server.fastmcp` и `mcp.run(transport="stdio")`).
> - `opencv-contrib-python` **4.14.x** — в 5.0 `WeChatQRCode` падает (нет InferenceEngine).
> - `pylibdmtx` **0.1.9** — в 0.1.10 модуль пустой.
> - Не ставьте `opencv-python` рядом с `opencv-contrib-python` — конфликт.
>   При конфликте: `pip install --force-reinstall --no-deps opencv-contrib-python`.

---

## 3. Запуск MCP-сервера

Сервер — это FastMCP-апп по **SSE** (HTTP). Он слушает `0.0.0.0:8000`
(эндпоинты `/sse` — GET для потока, `/messages/` — POST для JSON-RPC).
Весь лог идёт в `stderr` и в `mcp_server.log` (никогда не через `print`).

### 3.1. Прямой запуск (вручную)
```bash
cd /home/ps/photo-mcp-server
python3 photo_mcp_server.py
```
Вывод в stderr:
```
... photo-mcp-server ready (device=cuda, cv_backend=cpu)
Uvicorn running on http://0.0.0.0:8000
```
Сервер остаётся живым, ожидая MCP-клиента по SSE на `http://<host>:8000/sse`.
(В mcp 1.29.0 `host`/`port` задаются в конструкторе `FastMCP(...)`, а не в
`mcp.run(transport="sse")` — `run()` принимает только `transport`/`mount_path`.)

### 3.2. Подключение из MCP-клиента (Claude Desktop, IDE, свой клиент)
Сервер — сетевой (SSE), поэтому в конфигурации MCP-клиента указывают URL
(не stdio-команду):
```json
{
  "mcpServers": {
    "photo-mcp-server": {
      "url": "http://127.0.0.1:8000/sse"
    }
  }
}
```
Для удалённого хоста (например, vLLM Responses API `--tool-server`) замените
`127.0.0.1` на адрес сервера — он уже привязан к `0.0.0.0`.

### 3.3. Что происходит при старте
- Авто-выбор ускорения: **CUDA → MPS → CPU** (`_detect_device`).
- OpenCV DNN-бэкенд переключается на CUDA при наличии GPU (в 4.14 — предупреждение
  `setInferenceEngineBackendType` не смертельно, бэкенд падает на CPU).
- При **первом вызове** инструмента модели докачиваются:
  - WeChatQRCode → `wechat_models/` (detect/sr: `.prototxt` + `.caffemodel`).
  - YOLO → `yolov8n-barcode.pt`.
- Повторные запуски используют закэшированные модели (быстрый старт).

### 3.4. Инструменты

| Инструмент | Параметры | Возврат |
|-----------|-----------|---------|
| `detect_code` | `image_path` | `{"status":"success","detections":[{"bbox":[x1,y1,x2,y2],"confidence":0.99},...]}` |
| `align_perspective` | `image_path`, `corners` (4 точки `[[x,y],...]`), `output_size=300` | `{"status":"success","saved_path":"..."}` |
| `decode_code` | `image_path`, `bbox` (опц. ROI `[x1,y1,x2,y2]`) | `{"status":"success","results":[{"format":"QR","text":"..."},...]}` |

- `detect_code` — YOLO (`yolov8n-barcode.pt`) в `asyncio.to_thread`.
- `decode_code` — строгий каскад **WeChatQRCode → zxing-cpp → pylibdmtx → pyzbar**
  (стоп при первой успешной расшифровке).
- Все инструменты `async def`; блокирующий CV/ML-код — в `asyncio.to_thread`.
- Ошибки перехватываются: лог → `logger.exception`, возврат
  `{"status":"error","message":"..."}` — сервер никогда не падает.

---

## 4. Необходимые параметры vLLM

Модель обслуживается vLLM по OpenAI-совместимому API на `http://127.0.0.1:8001`.

**Требования к серверу модели (что нужно тесту):**
- OpenAI-совместимый эндпоинт `/v1/chat/completions` (GET `/v1/models`, `/health` — для проверки).
- **Поддержка tool/function calling** — модель должна возвращать `tool_calls`
  (Qwen3 поддерживает; vLLM применяет chat-шаблон).
- Достаточно большой контекст (в нашем случае `max_model_len ≈ 262144`).

**Проверка, что сервер жив:**
```bash
curl -s http://127.0.0.1:8001/v1/models      # -> {"data":[{"id":"Inferact/Qwen3.8-27B-NVFP4",...}]}
curl -s http://127.0.0.1:8001/health         # -> HTTP 200
```

**Пример запуска vLLM (адаптируйте под свою конфигурацию/квантизацию):**
```bash
vllm serve Inferact/Qwen3.8-27B-NVFP4 \
  --host 0.0.0.0 \
  --port 8001 \
  --max-model-len 262144 \
  --trust-remote-code
```
> Модель `Inferact/Qwen3.8-27B-NVFP4` — 27B Qwen3 в квантизации **NVFP4** (4-bit).
> Точный набор флагов (квантизация, память GPU, `--served-model-name`) — по вашей
> установке; критично, чтобы `/v1/chat/completions` отдавал `tool_calls`.
> Если модель лежит локально — замените HF-ID на путь к модели.

---

## 5. Тест-скрипт `test_mcp_model.py`

Тест — **MCP-агент**: модель на 8001 «мозг», `photo_mcp_server.py` — инструменты,
картинки из `test-imgs/` — вход. Для каждой картинки:
1. Запускается `photo_mcp_model.py` (SSE-сервер на `:8000`).
2. Клиент подключается по SSE (`/sse` → `/messages/`), делает JSON-RPC handshake
   (initialize) и получает список инструментов.
3. Схемы инструментов конвертируются в OpenAI function-call формат.
4. Модель получает промпт + инструменты и **сама** решает, какие вызвать.
5. Каждый `tool_call` исполняется на MCP-сервере, результат уходит обратно модели.
6. Модель выдаёт итоговый ответ (расшифрованные значения / «код не найден»).

> **Замечание:** тест использует **сырой JSON-RPC-клиент по SSE** (класс
> `RawSseClient` в скрипте), а не `mcp.ClientSession`. С этим FastMCP 1.29.0
> сервером `ClientSession` не завершает рукопожатие
> (`McpError: Invalid request parameters`), тогда как сырой JSON-RPC по SSE
> работает надёжно.

### 5.1. Запуск
```bash
cd /home/ps/photo-mcp-server
python3 test_mcp_model.py                 # первые 3 изображения
python3 test_mcp_model.py --limit 5       # первые 5
python3 test_mcp_model.py --all          # все из test-imgs/
```

### 5.2. Аргументы
| Аргумент | Значение по умолчанию | Описание |
|----------|----------------------|---------|
| `--model-url`  | `http://127.0.0.1:8001/v1/chat/completions` | Эндпоинт модели (OpenAI-совместимый). |
| `--model-name` | `Inferact/Qwen3.8-27B-NVFP4` | Имя модели. |
| `--limit N`    | `3` | Сколько изображений обработать. |
| `--all`        | — | Обработать все изображения в `test-imgs/`. |
| `--max-rounds` | `6` | Макс. раундов вызова инструментов на изображение. |

### 5.3. Пример вывода
```
Model : Inferact/Qwen3.8-27B-NVFP4 @ http://127.0.0.1:8001/v1/chat/completions
Images: 2/14 from /home/ps/photo-mcp-server/test-imgs
MCP tools available: ['detect_code', 'align_perspective', 'decode_code']

[1/2] 1000070476.jpg
  tool #1 detect_code({'image_path': '.../1000070476.jpg'}) [1.15s]
     -> {"status": "success", "detections": [{"bbox": [0.33, 619.53, 167.21, 712.0], "confidence": 0.755}]}
  tool #3 decode_code({'image_path': '.../1000070476.jpg'}) [6.01s]
     -> {"status": "success", "results": [{"format": "QR", "text": "ELK00000028956"}, {"format": "QR", "text": "0AGN3K3Y901930N"}]}
  RESULT: ... decoded two QR codes: ELK00000028956, 0AGN3K3Y901930N
```

---

## 6. Примеры промптов к модели

### 6.1. Промпт, который использует тест (`test_mcp_model.py`)

**System:**
```
You are a photo-fixation analysis agent for 1D/2D codes (DataMatrix, QR, barcodes).
You MUST use the provided tools to inspect the image; NEVER invent or guess a
code's content. Only report values that the tools actually return. If no code is
detected, say so clearly.
```

**User:**
```
Analyze this image: /home/ps/photo-mcp-server/test-imgs/1000070476.jpg
Use the available tools to detect any 1D/2D codes and decode them. Start with
detect_code, then decode_code (or align_perspective -> decode_code) as
appropriate. Report the decoded values, or state clearly that no code was found.
```

### 6.2. Дополнительные примеры промптов (ручной сценарий)

**Прямая расшифровка (без YOLO):**
```
Расшифруй все коды на картинке /path/to/img.jpg. Вызови decode_code и перечисли
все полученные значения. Если кода нет — так и скажи.
```

**Поиск области и затем расшифровка:**
```
Найди на картинке /path/to/img.jpg область с кодом через detect_code, затем
расшифруй её через decode_code (с bbox). Если bbox не расшифровывается —
расшифруй всю картинку целиком.
```

**Прямая перспективная коррекция по углам:**
```
На картинке /path/to/img.jpg код наклонён. Углы: [[10,10],[300,12],[320,300],[0,290]].
Вызови align_perspective и затем decode_code на сохранённой *_aligned.png.
```

**Жёсткая корректность (запрет догадок):**
```
Только то, что вернули инструменты. Если decode_code вернул пустой results —
напиши «код не обнаружен», не выдумывай содержимое.
```

---

## 7. Устранение неполадок

| Симптом | Причина / решение |
|---------|-------------------|
| `Failed to import FastMCP` | `mcp` должен быть **< 2**: `pip install "mcp==1.29.0"`. |
| `WeChatQRCode` SystemError / нет InferenceEngine | Поставьте `opencv-contrib-python==4.14.0.94` (не 5.0). |
| `pylibdmtx` нет `decode` | Поставьте `pylibdmtx==0.1.9` (не 0.1.10). |
| Конфликт `opencv-python`/`opencv-contrib-python` | `pip install --force-reinstall --no-deps opencv-contrib-python`. |
| `FastMCP.run() got an unexpected keyword argument 'host'` | В mcp 1.29.0 `host`/`port` задают в конструкторе `FastMCP(host=..., port=...)`, а не в `run(transport="sse")`. |
| `McpError: Invalid request parameters` (ClientSession) | Используйте сырой JSON-RPC по SSE (как в `test_mcp_model.py`), а не `mcp.ClientSession`. |
| Сервер «не отвечает» по SSE | Смотрите `mcp_server.log` / stderr; проверьте, что `Uvicorn running on http://0.0.0.0:8000`. |
| 404 при скачивании моделей | Проверьте интернет; модели кэшируются в `wechat_models/` и рядом с сервером. |

---

## 8. Безопасность
- Всё логирование — в `stderr` + `mcp_server.log`; **`print()`/stdout не используются**
  (stdout — канал MCP-протокола).
- Не храните токены в `.git/config`; при push по HTTPS используйте inline-токен
  и **отзовите его** после использования.
- `.gitignore` исключает `wechat_models/`, `*.pt`, `*.caffemodel`, `*.prototxt`,
  `*.log`, `__pycache__/`.