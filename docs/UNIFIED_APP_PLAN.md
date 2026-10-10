# План: одно десктопное приложение из WhisperSync и Podcast Reels Forge

> Исходный план от 2026-10-05; актуальная сверка реализации — 2026-10-10. Продукт — **Ultimate Video Forge**, публичная команда — `uvf`. Внутренний пакет `studio` сохранён для совместимости. Решения владельца в `docs/PROJECT_DECISIONS.md` имеют приоритет.
>
> Исходные проекты: [WhisperSync](https://github.com/Bormotoon/WhisperSync) (0.1.0, локально `/home/borm/VibeCoding/WhisperSync`) и Podcast Reels Forge (1.4.x, этот репозиторий).

## Оглавление

0. [Результаты аудита и обязательные изменения](#0-результаты-аудита-и-обязательные-изменения)
1. [Цель](#1-цель)
2. [Открытые решения](#2-открытые-решения)
3. [Исходное состояние](#3-исходное-состояние)
4. [Целевая архитектура](#4-целевая-архитектура)
5. [Этапы обработки](#5-этапы-обработки)
6. [Карта переноса: что, откуда, куда, как](#6-карта-переноса-что-откуда-куда-как)
7. [Фазы работ](#7-фазы-работ)
8. [Тестирование и приёмка](#8-тестирование-и-приёмка)
9. [Риски](#9-риски)
10. [Первые шаги](#10-первые-шаги)
11. [Приложения: форматы](#11-приложения-форматы)

---

## 0. Результаты аудита и обязательные изменения

Этот раздел добавлен после сверки плана с текущим кодом обоих проектов. Он имеет приоритет над более ранними формулировками ниже: если старый текст противоречит этому разделу, выполнять нужно этот вариант.

### 0.1. Что подтверждено исходниками

- На проверенных рабочих деревьях тесты проходят: WhisperSync — **403**, Forge — **620**. Это базовая точка миграции, а не доказательство готовности объединённого приложения.
- WhisperSync сейчас содержит незакоммиченные исправления в `engine/pipeline.py`, `engine/export.py`, `engine/media.py` и связанных тестах. `git subtree` импортирует только коммиты и эти исправления потеряет.
- В WhisperSync стратегия 3 (`Hybrid`) уже строит куски с разными коэффициентами, локальной регрессией и поглощением остатка в паузах. Это не сводится к одной формуле `offset + k·t`.
- Forge действительно переиспользует модели по крупным фазам (`whisper_model_session`, одна сессия llama на очередь), но многие стадии сейчас исполняются в процессе оркестратора. Формулировка «каждый этап уже отдельный процесс» неверна и должна быть целью миграции, а не предпосылкой.
- Анализ Forge использует транскрипт, диаризацию, метаданные, промпты, настройки и исходное аудио для признаков громкости/тишины. Поэтому его вход нельзя описывать только как `program.mp4` или JSON транскрипта.
- Кэш WhisperSync идентифицирует исходник по пути, размеру, `mtime`, аудиопотоку и параметрам декодирования; это хороший быстрый кэш, но не полноценная гарантия идентичности содержимого при сохранении `mtime`.
- Блокировки проектов различаются: WhisperSync использует переносимую lock-файл-схему, Forge — `flock` на POSIX и best-effort на Windows. Их нельзя механически слить без отдельного теста на Windows и сценариев аварийного завершения.
- Веб-редактор субтитров Forge сохраняет настройки через File System Access API браузера. Открытие HTML из `file://` и запуск через встроенный браузер — разные режимы; интеграционный контракт сохранения нужно определить заранее.

### 0.2. Критические проблемы первоначального плана

1. **Неполная модель времени.** `Placement` подходит для размещения исходного файла, но не для рендера синхронизированного голоса и не для результата Hybrid. Нужны отдельные `SourcePlacement`, кусочная `AudioWarpMap` и `EditMap`.
2. **Скрытый цикл planner → transcribe → sync → transcribe.** Авто-синхронизация может запросить транскрипцию камеры уже после решения планировщика. Это нужно заменить на явную фазу `discover/prepare` или на повторное планирование с версией входов.
3. **Неверный порядок program/reels.** Рилсы требуют либо существующий исходный видео-ассет, либо `program.mp4`. Нельзя безусловно ставить `program` после всех LLM-этапов и одновременно объявлять его обязательным входом `reels`.
4. **Слишком широкое обещание «ничего не пересчитывается».** Повторное использование возможно только при валидных, полных артефактах и отпечатке, включающем код/схему/модели/промпты/настройки и идентичность файлов.
5. **Недооценён перенос.** Таблицы «как есть» скрывают CLI-зависимости, глобальное состояние, пути к ресурсам, subprocess-контракты и модель ошибок. Сначала нужны characterization-тесты и фасады, затем перенос реализации.
6. **Опасные хуки и управление процессами.** Произвольные shell-хуки и остановка сервера по порту могут задеть чужой процесс. Нужны явные разрешения, PID/маркер владельца и безопасное поведение по умолчанию.
7. **Ранняя упаковка отсутствует.** PyInstaller, torch, Qt, ffmpeg, модели и модульные окружения должны проверяться до фазы «установщики», иначе архитектура может оказаться непакуемой.

### 0.3. Обязательные решения до написания нового ядра

- Зафиксировать commit/tag каждого источника и отдельно сохранить patchset текущего рабочего дерева WhisperSync. Не считать локальные изменения частью subtree, пока они не закоммичены или не оформлены как патч с проверяемым хэшем.
- Утвердить контракт артефактов: каждый JSON содержит `schema_version`, `producer_version`, `inputs`, `created_at`, `status`; незавершённый артефакт не считается результатом.
- Утвердить контракт времени: все значения имеют `TimeDomain` (`file`, `timeline`, `edited`, `rendered_audio`), а преобразования выполняются только через типизированные карты.
- Утвердить минимальный поддерживаемый набор: Linux + NVIDIA, Final Cut FCPXML и Premiere XMEML на фиксированных версиях. macOS/Windows smoke-тесты не заменяют импорт в NLE.
- Выбрать стратегию совместимости лицензий и провести проверку всех сторонних моделей, шрифтов, ffmpeg-сборок и pyannote/audio-separator до публикации нового репозитория.

### 0.4. Рекомендуемый MVP после аудита

Первый объединённый выпуск должен включать: сканирование, транскрипцию, существующую сложную синхронизацию WhisperSync, FCPXML, базовый CLI и проектные артефакты. `roughcut`, `program`, LLM, диаризация, рилсы и YouTube подключать последовательно после стабилизации контракта проекта. Это уменьшает площадь одновременной отладки и даёт рабочий продукт уже на ранней фазе.

---

## 1. Цель

### Что делает пользователь

1. Указывает папку, где лежат файлы с камер и рекордеров (можно вперемешку, можно по подпапкам). Вместо папки можно дать ссылку на YouTube.
2. Выбирает режим синхронизации:
   - **Авто** — по умолчанию;
   - **Звук камеры** — звук с камер нормальный, рекордер не нужен;
   - **Простая** — достаточно сдвига;
   - **Сложная** — нужна коррекция дрейфа.
3. Включает или выключает остальное: черновой монтаж, спикеров, рилсы, тексты. Выбирает, для кого проект: Final Cut, Premiere или оба.
4. Видит план: что найдено, какие этапы будут выполнены, какие пропущены и почему. Здесь же можно поправить ошибки распознавания (не та камера, не тот рекордер).
5. Нажимает «Пуск» и получает папку с результатом:
   - проект для Final Cut (`.fcpxml`) и/или Premiere (`.xml`);
   - синхронизированный звук;
   - транскрипт;
   - по желанию — черновой мастер, рилсы, вычитанный текст и статья.

### Принципы

- **Исходники неприкосновенны.** Программа ничего не пишет в файлы камер и рекордеров. Проекты для монтажных программ ссылаются на оригиналы и на отрендеренный звук.
- **Любой этап можно не выполнять.** Пропуск этапа не ломает остальные: каждый этап объявляет, что ему нужно на входе, и пропускается с понятной причиной, если входа нет.
- **Программа решает сама, пользователь может переопределить.** Планировщик выбирает этапы и режимы по содержимому папки и настройкам. Любое решение можно поменять вручную, и ручная правка сохраняется в проекте.
- **Ничего не пересчитывается зря.** У каждого этапа есть fingerprint и manifest с проверенными артефактами. Повторный запуск переиспользует только полностью опубликованный и валидный результат.
- **Всё локально.** Никаких облачных API: Whisper, pyannote и llama.cpp работают на машине пользователя.
- **Ядро не знает про интерфейс.** Вся логика без Qt — как сейчас у обоих проектов. Графический интерфейс и командная строка — две оболочки над одним ядром.

### Чего не делаем (по крайней мере в первой версии)

- Собственный таймлайн-редактор. Тонкий монтаж — в Final Cut или Premiere, а приложение готовит черновик.
- Цветокоррекцию, сведение звука, графику.
- Публикацию в соцсети.
- Облачные модели (провайдеры OpenAI/Anthropic/Gemini в `llm/providers.py` Forge остаются в коде, но в интерфейсе не показываются).

---

## 2. Открытые решения

Решения владельца от 2026-10-08 зафиксированы в
`docs/PROJECT_DECISIONS.md` и имеют приоритет над рекомендациями таблицы:
MIT, Ultimate Video Forge, разработка в `/srv/storage/docs/Ultimate-Video-Forge/`,
приёмка сначала на текущем Linux, исходные репозитории не изменяем,
редактор субтитров внутри существующего PyQt GUI (уточнение владельца от 2026-10-09).
Предпочитаем самые свежие стабильные
технологии после проверки совместимости; Python 3.12 остаётся целевой release-базой.
Текущие локальные тесты/сборка выполнены на Python 3.10.20; квалификация установленного
артефакта на 3.12 и переход на 3.14 требуют отдельных проверок.

План ниже исходит из рекомендованного варианта. Последний столбец — что изменится, если выбрать иначе.

| # | Вопрос | Рекомендация | Если выбрать иначе |
|---|---|---|---|
| 1 | Где строить | **Новый репозиторий.** Историю обоих проектов импортировать через `git subtree` (раздел 7, фаза 0). Основа интерфейса и CI — из WhisperSync. | «Выращивать из WhisperSync» — фаза 0 короче (импортируется только Forge), но имя `whispersync` останется в путях и истории, а переименование пакета всё равно понадобится. |
| 2 | Лицензия | Решать тебе. Юридически можно любую: WhisperSync целиком твой, а MIT разрешает включать код Forge. | PolyForm Noncommercial закрывает коммерческое использование. MIT открывает и код синхронизации. На код и план это не влияет, меняются только `LICENSE` и `pyproject.toml`. |
| 3 | Mac | **Синхронизация, транскрипт, черновой монтаж и экспорт — везде. Рилсы и всё на torch — только с NVIDIA.** | Полная поддержка Mac потребует Whisper через MLX и рендер без NVENC (VideoToolbox) — это отдельная фаза после 7. |
| 4 | Старые репозитории | **Заморозка функций сейчас, только исправления ошибок.** Архивировать WhisperSync после фазы 2, Forge — после фазы 5, когда новый догонит их по возможностям. | Если развивать параллельно, каждую фичу придётся переносить дважды. |
| 5 | Название | **Ultimate Video Forge**, distribution `ultimate-video-forge`, CLI `uvf`; внутренний `studio` сохраняется для совместимости. | Смена внутреннего namespace не требуется для публичного имени. |
| 6 | Редактор субтитров | **Внутри существующего PyQt GUI:** нативные Qt-контролы, видеопредпросмотр, сохранение через приложение. Решение владельца от 2026-10-09. | Веб-реализация Forge остаётся источником поведения и пресетов для переноса. |
| 7 | Версия Python | **3.12** для основного окружения. CI — 3.10–3.13. | 3.13+ — нужно сверить совместимость с torch и pyannote. |

---

## 3. Исходное состояние

| | WhisperSync | Podcast Reels Forge |
|---|---|---|
| Назначение | синхронизация звука рекордера с видео, дрейф часов, FCPXML | рилсы, вычитка и статья из длинного эпизода |
| Код / тесты, строк | 13 559 / 7 089 | 23 999 / 11 043 (+4 846 строк веб-интерфейса) |
| Тестов в проверенном рабочем дереве | 403 | 620 |
| Интерфейс | PyQt6 (тёмная тема, таймлайн, симулятор, справка) | CLI + статическая страница в браузере (сборка `config.yaml`, редактор субтитров) |
| Платформы | Windows, macOS, Linux; есть работа без GPU | фактически Linux + NVIDIA 16 ГБ |
| Тяжёлые зависимости | faster-whisper (без torch); `.sep-venv` на Python 3.12 под audio-separator | faster-whisper, torch, opencv, pyannote (опц.), llama-server, yt-dlp (опц.) |
| Настройки | dataclass `WhisperSyncConfig` + строгая проверка, JSON | `config.yaml` + `config.local.yaml`, словари |
| Лицензия | PolyForm Noncommercial 1.0.0 | MIT |
| Оркестрация | монолит `engine/pipeline.py` (2773 строки), один процесс + пул рендера | `pipeline.py` (1868) — оркестратор; часть ML-стадий вызывается in-process, отдельные CLI-модули запускаются subprocess-ами |

### Что уже общее

- **Настройки Whisper одинаковые:** их перенесли из Forge в WhisperSync. Это лестница температур, `repetition_penalty`, `no_repeat_ngram_size`, режимы `fast`/`quality`, OOM-лестница.
- **Формат транскрипта один:** WhisperSync уже пишет JSON «в формате Forge» (`engine/transcript_export.py` → `segments[].words[]` + `sentences`). На этом формате и строится общий контракт.

### Что дублируется и сольётся в одно

| Что | WhisperSync | Forge |
|---|---|---|
| Обёртка Whisper | `engine/transcriber.py` (класс `WhisperEngine`, кэш по SHA-256) | `stages/transcribe_stage.py`, `utils/subtitle_sync.load_model` |
| Проверка окружения | `engine/system_check.py` | `preflight.py`, `pipeline._has_cuda/_gpu_vram_gb` |
| Блокировка от двойного запуска | `engine/workspace.output_lock` | `utils/run_lock.py` |
| Запуск ffmpeg и пробы | `engine/media.py`, `engine/proc.py` | `utils/ffmpeg.py`, `utils/media_qa.py`, `analysis/audio_features.py` |
| Поиск пауз | `pipeline._silence_spans`, `pause_spans_local`, `recorder_word_gaps` | `analysis/audio_features.parse_silencedetect`, `utils/clip_intervals.speech_edges` |
| Логи | `logging_setup.py` | `utils/logging_utils.py` |

---

## 4. Целевая архитектура

### 4.1. Слои

```
┌──────────────────────────────┐   ┌──────────────────────────┐
│ gui/  (PyQt6)                │   │ cli/  (headless, batch)  │
└──────────────┬───────────────┘   └────────────┬─────────────┘
               │  события JSON-lines, команды    │
┌──────────────▼─────────────────────────────────▼─────────────┐
│ stages/planner.py   — что запускать и почему                  │
│ stages/runner.py    — процессы-исполнители, порядок по GPU,   │
│                       отпечатки, отмена, отчёт                │
└──────────────┬───────────────────────────────────────────────┘
               │ Stage API
┌──────────────▼───────────────────────────────────────────────┐
│ stages/{scan,transcribe,sync,timeline,speakers,roughcut,      │
│         text,export,program,reels,fetch}                      │
└──────┬─────────────────────┬─────────────────────┬───────────┘
┌──────▼──────┐   ┌──────────▼─────────┐   ┌──────▼──────────┐
│ core/       │   │ llm/  (llama.cpp)  │   │ vision/ (torch) │
│ модель      │   └────────────────────┘   └─────────────────┘
│ проекта,    │
│ медиа, время│   modules/ — менеджер опциональных окружений
└─────────────┘
```

Правила зависимостей:

- `core` ни от чего не зависит.
- `stages` зависят от `core`, а также от `llm` и `vision` (те — только внутри исполнителя).
- `gui` и `cli` зависят от `stages` (только от планировщика и исполнителя) и от `core`.
- В `core` и `stages` нет импортов Qt, torch и pyannote на уровне модуля. Тяжёлые импорты — внутри функций, которые выполняются в процессе-исполнителе.

### 4.2. Дерево пакетов

```
studio/
  __init__.py
  app.py                     # точка входа: GUI по умолчанию, --cli → cli
  core/
    project.py               # Project, Asset, Camera, Recorder, Placement, … + (де)сериализация project.json
    timeline.py              # Timeline, Track, Clip, KeepRange, Marker; преобразования времени
    transcript.py            # Word/Segment/Transcript + JSON-формат + SRT
    transcript_index.py      # пословный/пофразовый индекс (из Forge analysis/)
    transcript_align.py      # перенос таймингов на исправленный текст (из Forge utils/word_alignment)
    media.py                 # probe, извлечение звука (WhisperSync engine/media.py)
    ffmpeg.py                # выбор ffmpeg с NVENC/libass, аргументы кодека (Forge utils/ffmpeg.py)
    media_qa.py              # проверка готового ролика (Forge)
    audio_spans.py           # паузы/тишина/энергия: общее для sync, roughcut, reels
    proc.py                  # запуск долгих процессов (WhisperSync)
    workspace.py             # атомарная публикация, блокировка папки, временные каталоги
    fingerprint.py           # отпечатки входов (Forge)
    naming.py, sources.py    # естественный порядок имён, стабильные id источников (WhisperSync)
    settings/                # модель настроек по разделам + загрузка YAML + проверка
    capabilities.py          # CUDA, VRAM, NVENC, llama-server, модули, ffmpeg
    env.py                   # .env с токенами (HF, YouTube)
    logging.py
    report.py                # отчёт о прогоне (Forge run_report)
  stages/
    base.py                  # Stage, Decision, Requirement, StageContext, StageOutput
    planner.py
    runner.py
    worker.py                # точка входа процесса-исполнителя: python -m studio.stages.worker
    events.py                # протокол событий JSON-lines
    scan/       transcribe/   sync/        timeline/   speakers/
    roughcut/   text/         export/      program/    reels/      fetch/
  llm/
    providers.py, schemas.py, json_utils.py
    server.py                # запуск/остановка llama-server (Forge utils/llama_cpp_service.py)
    session.py               # одна сессия сервера на группу LLM-этапов (Forge pipeline._LlamaSession)
  vision/
    face_crop.py, face_track.py, active_speaker.py
  modules/
    manager.py               # установка/проверка опциональных окружений
    recipes/                 # что ставить для separation, enhance, vision, diarization, llm, youtube
  gui/
    main_window.py
    screens/                 # start, material, plan, run, result, settings, help
    widgets/                 # drop_zone, log_view, timeline_preview, sync_simulator, …
    bridge.py                # QProcess ↔ runner, события → сигналы
    theme.qss
  cli/
    main.py                  # studio scan|plan|run|export|doctor|verify|batch|modules
    batch.py                 # прогон без присмотра (Forge autonomy)
  resources/
    prompts/{ru,en}/  fonts/  models/  subtitle_editor/  help/
tests/  (зеркалит studio/)
contrib/
  hooks/host_memory/         # освобождение памяти виртуалок (Forge host_memory) как хук
tools/
  evaluate_prompts.py  make_fixtures.py  migrate_settings.py
packaging/
  pyinstaller/  ci/
docs/
```

### 4.3. Время: четыре шкалы и две карты

Это главное, на чём легко ошибиться. Все этапы обязаны явно знать, в какой шкале у них числа.

| Шкала | Что это | Кто в ней живёт |
|---|---|---|
| **Время файла** (asset time) | секунды от начала конкретного файла | транскрипты источников, кэш Whisper, якоря синхронизации |
| **Время таймлайна** (timeline time) | общая шкала съёмки; ноль — начало самого раннего материала, опорная шкала — часы главной камеры (как `timebase_source: camera` в WhisperSync) | размещение файлов, транскрипт таймлайна, спикеры, решения монтажа, экспорт |
| **Время монтажа** (edited time) | шкала после вырезания лишнего | черновой мастер, транскрипт мастера, рилсы |
| **Время рендера** (rendered-audio time) | локальная шкала конкретного WAV после time-stretch/resample | `SyncedAudio`, self-check, экспорт звука |

Преобразования:

- **файл → таймлайн:** `SourcePlacement` — размещение исходного файла. Для файла допускается affine map `offset + k·t`, но это не описывает рендер голоса.
- **рекордер → rendered-audio:** `AudioWarpMap` — упорядоченные куски `(source_start, source_duration, output_start, output_duration, method)`. Она должна точно покрывать выходной диапазон без дыр и перекрытий; это модель для WhisperSync strategy 2/3, pause absorption и self-check.
- **таймлайн → монтаж:** `EditMap`/`EditList` — упорядоченный список оставленных диапазонов. Время монтажа = сумма длин предыдущих диапазонов + смещение внутри текущего. Для удалённых диапазонов обратного преобразования нет, функция должна возвращать `None`.
- `core/timeline.py` даёт типизированные функции `file_to_timeline`, `timeline_to_file`, `audio_source_to_rendered`, `rendered_to_audio_source`, `timeline_to_edited`, `edited_to_timeline` и `map_words`. Этапам запрещена ручная арифметика времени.
- Для каждого артефакта хранится `time_domain` и `map_id`. Тесты обязаны проверять монотонность, покрытие, round-trip на сохраняемых диапазонах, границы, отрицательную A/V-калибровку, VFR и кусочный Hybrid.

### 4.4. Модель проекта

Набросок. Это dataclass'ы в `core/project.py`, сериализация в `project.json` с полями `version` и `schema_version` для миграций.

```python
@dataclass
class Asset:                      # один файл из папки
    id: str                       # стабильный slug (WhisperSync sources.assign_source_ids)
    path: Path                    # относительно корня папки, если внутри неё
    kind: Literal["video", "audio", "other"]
    role: Literal["camera", "recorder", "ignore", "unknown"]
    device: str | None            # «DJI Osmo Pocket 3», «ZOOM H6» — из метаданных или имени
    group_id: str | None          # к какой камере/рекордеру относится
    chapter_of: str | None        # продолжение записи (главы GoPro, разбиение по 4 ГБ)
    info: MediaInfo               # результат probe (WhisperSync media.MediaInfo)
    created_at: datetime | None
    timecode: str | None
    manual: dict[str, Any]        # ручные правки из интерфейса, переживают пересканирование

@dataclass
class Camera:   id: str; name: str; asset_ids: list[str]; speaker: str | None; is_wide: bool | None
@dataclass
class Recorder: id: str; name: str; asset_ids: list[str]; tracks: list[RecorderTrack]
@dataclass
class RecorderTrack: index: int; channels: list[int]; speaker: str | None   # дорожка многодорожечного рекордера

@dataclass
class Placement:                  # где файл лежит на таймлайне
    asset_id: str
    offset_s: float; in_s: float; duration_s: float; k: float = 1.0
    provenance: Literal["filename", "metadata", "timecode", "acoustic", "text", "repair", "manual"]
    evidence: dict[str, float]    # inliers, residual_ms, coverage — как AlignmentMap.evidence

@dataclass
class AudioWarpPiece:
    source_start_s: float; source_duration_s: float
    rendered_start_s: float; rendered_duration_s: float
    factor: float; method: Literal["copy", "atempo", "resample"]

@dataclass
class AudioWarpMap:
    source_asset_id: str; target_asset_id: str | None
    pieces: list[AudioWarpPiece]
    source_domain: Literal["file", "timeline"]
    rendered_duration_s: float
    strategy: int; evidence: dict[str, float]

@dataclass
class SyncedAudio:                # отрендеренный голос под конкретный файл камеры или весь таймлайн
    path: Path; for_asset: str | None; source_ref: tuple[str, float, float]
    warp_map_id: str; time_domain: Literal["rendered_audio"] = "rendered_audio"

@dataclass
class SpeakerTurn: start: float; end: float; speaker: str; source: Literal["mics", "pyannote", "manual"]

@dataclass
class KeepRange:                  # кусок, который остаётся в монтаже (время таймлайна)
    start: float; end: float; camera_id: str | None

@dataclass
class Cut:                        # что убрано и почему — для маркеров и проверки
    start: float; end: float
    reason: Literal["pause", "retake", "head", "tail", "filler", "llm", "manual"]
    confidence: float; note: str; accepted: bool = True

@dataclass
class EditList: keep: list[KeepRange]; cuts: list[Cut]; markers: list[Marker]; mode: Literal["cut", "markers"]

@dataclass
class Project:
    version: int
    source_dir: Path; work_dir: Path
    assets: list[Asset]; cameras: list[Camera]; recorders: list[Recorder]
    placements: list[Placement]; synced_audio: list[SyncedAudio]
    transcripts: dict[str, Path]          # asset_id | "timeline" | "edited" → JSON
    speakers: list[SpeakerTurn]; speaker_names: dict[str, str]
    edit: EditList | None
    outputs: dict[str, list[Path]]        # export, program, reels, text
    stage_states: dict[str, StageState]   # отпечатки (Forge fingerprint.StageState)
    schema_version: int = 1
```

`SyncPlan` и `MediaClip` из WhisperSync перестают быть внешним форматом. В новой модели их заменяют `SourcePlacement`, `AudioWarpMap`, `SyncedAudio`, `KeepRange` и `Timeline` (раздел 6). Внутри compatibility facade их можно оставить на время переноса.

### 4.5. Папка проекта

По умолчанию — `<папка с материалом>/_studio/`: пути получаются относительными, и проект переносится вместе с материалом. Если папка только для чтения, рабочая папка задаётся отдельно.

```
_studio/
  project.json              # модель проекта — единственный источник правды
  settings.yaml             # настройки ЭТОГО проекта (только отличия от общих)
  stages/
    scan/report.json
     prepare/requirements.json, plan-inputs.json
    transcribe/<asset_id>.json, .srt
    sync/voice/<asset_id>_voice.wav, ambience/…, segments/…, verify.json, self_check.json
    timeline/transcript.json
    speakers/diarization.json   # формат Forge: [{start, end, speaker}]
    roughcut/edit.json
    text/proofread.json, article.md
    reels/moments.json, reels_summary.md
  program/program.mp4, program.transcript.json
  export/<имя>.fcpxml, <имя>.xml, transcripts/, <имя>_master.wav
  reels/reel_01.mp4 …
   logs/run-<дата>.log, report.json
   manifests/<stage>.json             # schema, inputs, status, checksums
```

Кэш Whisper — не в проекте, а в общем пользовательском каталоге кэша (`platformdirs`, как `cache_dir` в WhisperSync). Тогда тот же материал в другом проекте не распознаётся заново.

### 4.6. Этап: интерфейс, планировщик, исполнитель

```python
class Stage(Protocol):
    id: str                                  # "sync"
    title: str                               # «Синхронизация» (для интерфейса)
    after: tuple[str, ...]                   # от каких этапов зависит
    gpu: GpuUse                              # NONE | WHISPER | TORCH | LLM | NVENC

    def requirements(self, s: Settings) -> list[Requirement]: ...
        # Requirement(kind="cuda"|"vram_gb"|"module"|"llama"|"nvenc"|"ffmpeg_libass",
        #             value=..., optional=bool, why="…")
    def decide(self, p: Project, s: Settings, caps: Capabilities) -> Decision: ...
        # Decision.run(choices={...}) | Decision.skip(reason) | Decision.blocked(reason, fix)
    def fingerprint(self, p: Project, s: Settings) -> str: ...
    def run(self, ctx: StageContext) -> StageOutput: ...
        # ctx: project (копия), settings, workdir, progress(v, msg), warn(msg),
        #      cancelled() -> bool, emit_timeline(snapshot), log
```

**Планировщик** (`planner.py`):

1. Выполняет двухфазное планирование. Сначала `discover/scan/prepare` определяют полный набор входов и требований. Только после этого строится DAG вычислений. Никакая стадия не может молча добавить новый вход уже после построения плана.
2. Если этап пропущен или заблокирован, зависимые этапы получают `skip` с причиной «нет входа X». Исключение — этапы, у которых этот вход необязателен (например, `export` без `roughcut` экспортирует таймлайн без вырезок).
3. Учитывает ручные переопределения (`--only`, `--skip`, галочки в интерфейсе) и показывает их в плане как «выключено пользователем».
4. Возвращает `Plan`: список `PlannedStage(stage, decision, choices, will_reuse: bool)`, где `will_reuse` означает: manifest существует, имеет статус `ok`, все перечисленные артефакты существуют и fingerprint совпал.
5. Если `sync.auto` после `prepare` требует дополнительных транскриптов, planner строит новую ревизию плана (`plan_revision=2`) и показывает добавленные входы. Циклическая зависимость запрещена.

**Исполнитель** (`runner.py`):

- Каждый тяжёлый или недоверенный этап с `decision = run` запускается **отдельным процессом**: `python -m studio.stages.worker <stage_id> <project.json> <settings.yaml>`. Лёгкие чистые этапы могут исполняться in-process только после отдельного измерения. Если этапу нужен модуль, используется интерпретатор модуля. Зачем отдельный процесс:
  - видеопамять гарантированно освобождается после этапа;
  - падение этапа не роняет интерфейс;
  - отмена работает всегда (так уже устроен Forge, `pipeline.run_module`);
  - этапы на torch можно запускать интерпретатором окружения `vision`.
- Исполнитель читает из stdout процесса события JSON-lines (раздел 11.2) и пересылает их в интерфейс или в CLI.
- После успеха этап отдаёт `StageOutput`: изменения в модели проекта и список файлов. Исполнитель сначала проверяет manifest и артефакты, затем атомарно публикует результат и обновляет `project.json`. Статус `ok` записывается последним.
- **Ошибки.** Упавший этап помечается `failed`, зависимые — `skip(«упал X»)`. Независимые этапы продолжают работу: упала статья — экспорт всё равно делается. Это поведение Forge «упавший эпизод не роняет очередь», перенесённое на этапы.
- **Отмена:** флаг-файл/событие в `workdir` (кооперативная отмена), затем `terminate` через N секунд и остановка только дерева процессов данного этапа. Модель проекта не меняется, частичные файлы удаляются (`RunWorkspace` из WhisperSync).

### 4.7. GPU и память

- Этапы с GPU идут строго по одному.
- Порядок выбран так, чтобы тяжёлое не грузилось дважды. Он совпадает с нынешним порядком Forge «Whisper → pyannote → llama → NVENC»:
  ```
  scan → prepare → transcribe → sync → timeline → speakers → roughcut
       ├──────────────────────────────────────────────────────────────→ export
       └──────────────────────────────────────────────────────────────→ program
  program/timeline → [proofread → article → reels-select] → reels-render
                    └──────────── одна сессия llama-server ───────────┘
  ```
- `program` не обязателен для готового одиночного видео: `reels-render` может использовать исходный asset с явной картой времени. Для смонтированного master он зависит от `program`.
- **llama-server** поднимается один раз на группу LLM-этапов и гасится после неё (`llm/session.py`, из Forge `_LlamaSession`). Модель, кэш ответов и автонастройка контекста — как в Forge (`_autotune_llama_cpp_conf`, `CachingProvider`).
- **Хуки.** В `settings.local.yaml` задаются `hooks.*`, но по умолчанию они выключены. Команда запускается без shell-строки, через allowlist/явное подтверждение; сохраняются PID, владелец и результат. Запрещено останавливать процесс только по порту или `pkill -f`; llama-сессия завершает только свой PID/process group.
- По правилу «тяжёлое — только на GPU»: если CUDA нет, этапы с `gpu ≠ NONE` по умолчанию получают `blocked(«нет CUDA»)`. Работа на процессоре включается отдельной настройкой `compute.allow_cpu` (для Mac и машин без NVIDIA).

### 4.8. Настройки

- **Модель:** dataclass'ы по разделам (`ScanSettings`, `SyncSettings`, `TranscribeSettings`, `SpeakerSettings`, `RoughcutSettings`, `ExportSettings`, `ProgramSettings`, `TextSettings`, `ReelsSettings`, `LlmSettings`, `HooksSettings`, `BatchSettings`, `FetchSettings`).
- **Проверка** — подход WhisperSync (`config.py`): типы, перечисления, диапазоны, конечные числа. Проверяется один раз после слияния всех слоёв. Ошибка — одна строка и код возврата 2, а не странный результат через час.
- **Формат файла** — YAML, как у Forge (человекочитаемый, с комментариями).
- **Слои** (каждый следующий перекрывает предыдущий):
  1. умолчания в коде;
  2. общий `settings.yaml` пользователя (каталог настроек `platformdirs`);
  3. `settings.local.yaml` рядом с ним — машинные пути, порты, хуки (идея `config.local.yaml` Forge);
  4. `_studio/settings.yaml` проекта;
  5. `--set раздел.ключ=значение` из CLI или правки в интерфейсе.
- **Пресеты** — готовые наборы: «Подкаст: 2 человека, 2 камеры, петлички», «Соло на камеру», «Интервью с рекордером», «Готовое видео с YouTube».
- **Миграция:** `tools/migrate_settings.py` переводит старый `config.yaml` Forge и JSON-конфиг WhisperSync в новые разделы (таблица соответствия — раздел 6.3).

Разделы (сокращённо):

```yaml
scan:       {ignore: [...], grouping: auto|folders|names, device_patterns: {...}}
sync:       {mode: auto|camera|simple|complex, strategy: 1|2|3, auto: {max_drift_ms: 20},
             camera_audio_source: auto|<camera>, boundary_flex: true, seam_snap_max_s: 0.4,
             crossfade_ms: 10, pause_duck: {...}, ambience: {...}, voice_enhance: off,
             self_check: off|warn|repair, verify: false, camera_av_offset_ms: {...},
             voice_segment_minutes: 0, master_wav: false}
transcribe: {model: large-v3, language: ru, device: auto, compute_type: auto, mode: fast,
             batch_size: 16, initial_prompt: "", glossary: [...], keep_fillers: true}
speakers:   {method: auto|mics|pyannote|off, num_speakers: null, names: {...}}
roughcut:   {enabled: true, mode: cut|markers, pauses: {...}, retakes: {...}, head_tail: {...},
             fillers: {...}, multicam: {...}, llm: {enabled: false, ...}}
export:     {targets: [fcpxml, xmeml], fcpxml_version: "1.9", name: "", relative_paths: true}
program:    {enabled: auto, encoder: nvenc, quality: high}
text:       {proofread: {...}, article: {...}}
reels:      {...}   # нынешние processing/subtitles/video/exports/audio/prompts из Forge
llm:        {server: {...}, model: gemma4, cache: true}
hooks:      {before_run: "", after_run: "", before_gpu: "", after_gpu: ""}
batch:      {...}   # нынешний autonomy из Forge
fetch:      {...}   # нынешний youtube из Forge
compute:    {allow_cpu: false}
```

### 4.9. Опциональные модули

Базовая установка: PyQt6, faster-whisper/ctranslate2, numpy, PyYAML, Pillow, fonttools, platformdirs. Всё тяжёлое — в отдельные окружения, которые ставит `modules/manager.py`. Принцип уже работает в WhisperSync (`.sep-venv` и `setup_sep_venv.sh`).

| Модуль | Что внутри | Нужен этапам | Откуда рецепт |
|---|---|---|---|
| `separation` | audio-separator (Python 3.12) | sync → эмбиенс, enhance → denoise/dereverb | `setup_sep_venv.sh`, `requirements-sep.txt` |
| `enhance` | Resemble Enhance и др. | sync → voice_enhance | `engine/enhance.py` (`resemble_package_dir`) |
| `vision` | torch, opencv, модели YuNet/Light-ASD | reels-render (кадр за говорящим), roughcut → авто-привязка камер | `requirements.txt` Forge |
| `diarization` | pyannote.audio + torch, токен HF | speakers → pyannote | extras `diarization` Forge |
| `llm` | бинарник llama-server (CUDA/Metal) + GGUF gemma4 | text, roughcut-LLM, reels-select | `llama_cpp_service.py`, `config.yaml → llama_cpp` |
| `youtube` | yt-dlp | fetch | extras `youtube` Forge |

Код этапа, запущенный в окружении модуля, — это тот же пакет `studio`: менеджер кладёт путь к нему в `PYTHONPATH` интерпретатора модуля. Поэтому `core` должен работать на всех версиях Python, которые используют модули (3.12 — общий знаменатель).

В фазах 1–5 разработка идёт в одном окружении, как сейчас (`whisper-env` с torch). Менеджер модулей появляется в фазе 6 вместе с упаковкой.

---

## 5. Этапы обработки

Для каждого этапа: когда запускается, вход, выход, откуда берётся код, что пишется заново, на что обратить внимание.

### 5.0. `fetch` — альтернативный вход: YouTube

- **Когда:** пользователь дал ссылку, а не папку.
- **Выход:** скачанный файл и `.info.json` в `_studio/`. Проект из одного ассета `camera` со звуком. В метаданных отмечается, что материал уже смонтирован, поэтому планировщик сам пропускает `sync` и `roughcut`.
- **Код:** Forge `stages/fetch_stage.py`, `sources/youtube.py`, `sources/episode_metadata.py`, `scripts/fetch_youtube.py` — **как есть**. Меняются только точка входа (`run(ctx)` вместо CLI) и путь выгрузки.
- **Заметки:** перечисление канала или плейлиста — это режим `batch` (несколько проектов), а не один проект.

### 5.1. `scan` — разбор папки (новое)

- **Когда:** всегда.
- **Вход:** путь к папке и ручные правки из прошлого скана.
- **Выход:** `assets`, `cameras`, `recorders`, предупреждения, сводка для экрана плана.
- **Как:**
  1. Рекурсивный обход. Пропускаются скрытые файлы, сама `_studio/` и служебные файлы камер: `.LRF`/`.LRV` (прокси), `.THM`, `.SRT`-телеметрия DJI. Список настраивается (`scan.ignore`).
  2. Параллельный `probe` (WhisperSync `core/media.probe`): есть ли видео (не обложка), аудио, каналы, длительность, fps, `creation_time`, таймкод, теги производителя и модели (`com.apple.quicktime.make/model`, `encoder`, `handler_name`).
  3. **Роль.** Есть видеопоток → `camera`. Только звук → `recorder`. Не медиа → `ignore`. Короткий звук, который не удалось разместить на этапе sync (джингл, музыка), потом предлагается пометить как `ignore`.
  4. **Группировка по устройствам, по приоритету:**
     - подпапки (нынешнее соглашение WhisperSync «каждая камера в своей подпапке»);
     - теги производителя и модели;
     - шаблоны имён.
     Шаблоны — таблица в настройках. Примеры ниже **проверить на реальных файлах**:

     | Устройство | Пример имени | Заметки |
     |---|---|---|
     | DJI (Osmo, Action, Pocket) | `DJI_0838.MP4`, `DJI_20251004120000_0001_D.MP4` | рядом `.LRF`-прокси |
     | GoPro | `GX010024.MP4`, `GX020024.MP4` | вторая и третья цифры — номер главы одной записи |
     | Sony | `C0001.MP4` + `C0001M01.XML` | XML-спутник: модель и таймкод |
     | Canon | `MVI_1234.MP4` | |
     | iPhone | `IMG_1234.MOV` | тег `make = Apple` |
     | Zoom (H-серия) | `ZOOM0001.WAV`, `ZOOM0001_Tr1.WAV`, `ZOOM0001_LR.WAV` | дорожки одного дубля — один рекордер, общие часы |
     | DJI Mic, Rode, Tascam | `DJI_*.WAV`, `TASCAM_*.wav` и т. п. | достаточно «только звук» |

  5. **Главы и разбиение.** Файлы одной камеры, у которых `created_at + duration ≈ created_at` следующего или номера идут подряд в одной записи (главы GoPro), склеиваются в непрерывную последовательность без зазора. Здесь используется `naming.py` WhisperSync (естественный порядок, поиск последовательных серий).
  6. **Дорожки рекордера.** Многоканальный WAV или набор `_Tr1`, `_Tr2` одного дубля превращается в `RecorderTrack` со своим `speaker = None`. Имена спикеров даются позже.
  7. **Предупреждения:** «в видео нет звука — по звуку не синхронизировать, нужен таймкод или ручной сдвиг», «файл не опознан», «две камеры с одинаковыми именами файлов».
- **Откуда:** новый код. Используются `scan_cameras`, `scan_video_clips`, `preliminary_offsets`, `sequence_order_warnings` из `engine/pipeline.py` WhisperSync (строки 141–322) и `find_input_queue`, `_ensure_audio_companions` из Forge `pipeline.py`.
- **В интерфейсе:** экран «Материал» — дерево «устройство → файлы» с миниатюрами. Файл можно перетащить в другую группу, поменять роль, назвать камеру или дорожку. Правки пишутся в `Asset.manual` и переживают повторный скан.

### 5.2. `transcribe` — распознавание источников

- **Когда:** почти всегда. **Что именно** распознавать, решает `decide()`:
  - звук рекордера (каждый дубль; у многодорожечного — сведение или лучшая дорожка, см. 5.5) — всегда, если рекордер есть;
  - звук камер — только если синхронизации нужны текстовые якоря (режимы `complex` или `auto`, когда акустика не справилась) или если рекордера нет.
- **Выход:** `stages/transcribe/<asset_id>.json` и `.srt` во **времени файла**.
- **Откуда:**
  - основа — `WhisperEngine` из WhisperSync `engine/transcriber.py`: кэш по SHA-256 с учётом фактического устройства, OOM-лестница, батчи, выгрузка модели;
  - из Forge `stages/transcribe_stage.py`: разбиение `.srt` по предложениям (`_segment_to_srt_cues`, `_wrap_on_words`, `_build_sentence_groups`), `confidence`/`avg_logprob` сегментов, сессия модели на всю очередь (`whisper_model_session`);
  - сериализацию объединить в `core/transcript.py` на основе `engine/transcript_export.py` WhisperSync, в котором уже формат Forge.
- **Новое:**
  - `transcribe.glossary` — имена и термины из настроек (у Forge глоссарий сейчас есть только у вычитки). Передаётся в `hotwords` или `initial_prompt`.
  - `transcribe.keep_fillers`. Whisper склонен выбрасывать «ээ» и «мм», а черновому монтажу они нужны. Пробуем `initial_prompt` с примерами («Ээ, ну, мм, как бы…»). Эффект измеряем на реальной записи (риск в разделе 9).
- **GPU:** WHISPER. Модель загружается один раз на весь этап, все файлы — в одном процессе.

### 5.2a. `prepare` — фиксация полного набора входов

- `prepare` выполняется после `scan` и до построения основного плана. Он решает, какие дорожки, камеры и аудиофайлы нужны для первичного размещения, `sync.auto`, спикеров и рилсов.
- Результат — `requirements.json`: список `asset_id`, аудиопотоков, требуемых транскриптов, модулей и причин выбора. Он является входом для fingerprints и UI-плана.
- Если акустика/метаданные не дают решения, `prepare` заранее добавляет транскрипцию камеры в требования. Повторный запуск planner не должен запускать транскрипцию из середины `sync`.
- Любое ручное изменение групп, ролей или источника аудио увеличивает `plan_revision` и инвалидирует только зависимые стадии.

### 5.3. `sync` — размещение и синхронизация

Синхронизация решает **две разные задачи**, и их стоит развести:

1. **Размещение** — где каждый файл лежит на общей шкале. Нужно всегда, когда источников больше одного, в том числе если камер две, а рекордера нет.
2. **Подгонка** — заставить звук рекордера идти вровень с картинкой на всей длине (дрейф часов). Нужна только при режиме, отличном от «звук камеры».

#### Режимы

| Режим | Размещение | Подгонка | Рендер звука |
|---|---|---|---|
| `camera` («звук камеры») | камеры относительно друг друга: акустика, при неудаче текст | нет | нет; в экспорте звук камеры (`sync.camera_audio_source`) |
| `simple` | акустический сдвиг (GCC-PHAT), при неудаче текстовые якоря | один коэффициент на клип (стратегия 1 «Global Linear») | **нет, если `k ≈ 1`**: экспорт ссылается на оригинальный файл рекордера со сдвигом; иначе передискретизация целиком |
| `complex` | текстовые якоря + RANSAC (как сейчас) | стратегия 3 «Hybrid» (2 — в расширенных настройках) | да, кусками (как сейчас) |
| `auto` | как `simple`, плюс замер дрейфа | выбирается по замеру | по выбранному |

**Правило для `auto`:**

1. Для каждой пары «клип камеры ↔ рекордер» мерим сдвиг акустически в нескольких окнах по длине клипа (`acoustic_coarse_align` / `gcc_phat` из `engine/acoustic.py`) и строим прямую.
2. `|k − 1| · длительность < sync.auto.max_drift_ms` (по умолчанию 20 мс — половина кадра при 25 к/с) → `simple` без рендера.
3. Дрейф линейный (остаток мал, локальные `k` стабильны) → стратегия 1.
4. Иначе → `complex`. Решение принимает уже существующий `matcher.recommend_strategy`.
  5. Если акустика не дала уверенного результата, нужная транскрипция камеры должна быть уже добавлена этапом `prepare`. `sync` не вызывает другой stage и не меняет граф вычислений.

#### Что входит в этап, откуда и как

Монолит `_run_pipeline_locked` (`engine/pipeline.py`, 1500–2773) режется по шагам, которые там уже идут подряд:

| Шаг | Строки в `pipeline.py` WhisperSync | Куда | Как |
|---|---|---|---|
| Сканирование камер | ~1540–1617 | `stages/scan` | заменяется новым сканером |
| Транскрибация рекордеров и камер | ~1623–1729 | `stages/transcribe` | выносится в отдельный этап |
| Выравнивание (якоря, RANSAC, акустический запасной путь, проверка карты) | ~1730–1818, `_try_acoustic_fallback`, `_anchor_count`, `_map_evidence` | `stages/sync/placement.py` | функции как есть, обвязка заново |
| План стратегии, куски, привязка швов к тишине, Boundary Flex | ~1874–2097; функции 324–935 | `stages/sync/pieces.py` | как есть |
| Дубли → auditions/маркеры | внутри планирования, `_retake_groups_for_clip`, `_retake_groups_in_segment` | `stages/roughcut` | **переезжают в черновой монтаж** |
| Рендер кусков, пул процессов, сборка | ~2112–2296; функции 936–1160 | `stages/sync/render.py` | как есть |
| Улучшение голоса | ~2297–2346 | `stages/sync/enhance_step.py` | как есть, модуль `enhance` |
| Самопроверка и ремонт | ~2347–2425, `_repair_span` | `stages/sync/self_check_step.py` | как есть |
| Нарезка голоса на N-минутные сегменты | ~2426–2501, `_quiet_cut_points` | `stages/sync/segments.py` | как есть |
| Эмбиенс (отделение голоса камеры) | ~2517–2663 | `stages/sync/ambience_step.py` | как есть, модуль `separation` |
| Мастер-WAV | ~2664–2690 | `stages/export` | переезжает в экспорт |
| FCPXML | ~2691–2746 | `stages/export` | переезжает в экспорт |

- **Выход:** `SourcePlacement`, `AudioWarpMap`, `SyncedAudio`, отчёты `verify.json` и `self_check.json`, предупреждения (несопоставленные клипы, большой остаток, совет по стратегии).
- **Прогресс таймлайна** (`PipelineProgress`, `make_timeline_entries`) превращается в событие `timeline` протокола (раздел 11.2). Виджет `timeline_preview.py` получает его как сейчас.
- **GPU:** NONE для выравнивания и рендера (процессор, пул). WHISPER — для самопроверки. TORCH (в окружении модуля) — для эмбиенса и улучшения голоса.

### 5.4. `timeline` — транскрипт таймлайна (новое, лёгкое)

- **Когда:** всегда после `sync` или сразу после `transcribe`, если синхронизировать нечего.
- **Как:** берём транскрипт «главного» звука (рекордер; без него — выбранная камера) и пересчитываем слова через `Placement` во время таймлайна (`core/timeline.map_words`). Если рекордеров несколько и включён `recorder_mode: best`, на каждый отрезок берётся лучший, как сейчас при выборе рекордера на клип.
- **Выход:** `stages/timeline/transcript.json` — **без повторного Whisper**.

### 5.5. `speakers` — кто говорит

- **Когда:** `speakers.method ≠ off`. При `auto`:
  - у рекордера ≥ 2 дорожек со спикерами (или ≥ 2 рекордеров в режиме `all`) → `mics`;
  - иначе, если установлен модуль `diarization` и людей явно больше одного → `pyannote`;
  - иначе пропуск.
- **`mics` (новое):**
  1. Огибающая громкости (RMS, шаг 50 мс) каждой синхронизированной дорожки на шкале таймлайна.
  2. Для каждого слова транскрипта спикер — дорожка с наибольшей средней энергией, если она громче второй на `≥ 6 дБ`. Иначе — `overlap`.
  3. Сглаживание: реплики короче порога сливаются с соседними. Идея — `merge_short_turns` из Forge `face_track.py`.
  4. Выход в формате Forge `diarization.json` (`[{start, end, speaker}]`) во времени таймлайна. Модуль не нужен, GPU не нужен.
  5. Звук из чужих петличек (bleed) при этом не мешает: слова берутся из одного транскрипта, а дорожки служат только для сравнения громкости.
- **`pyannote`:** Forge `scripts/diarize.py` → `stages/speakers/pyannote.py`. `main(argv)` превращается в функцию, запускается в окружении `diarization`, параметр `num_speakers` сохраняется.
- **Реплики:** Forge `analysis/speaker_turns.py` → `stages/speakers/turns.py`, **как есть**. Им пользуются `text`, `reels` и `roughcut`.
- **Имена:** интерфейс («Дорожка 1 = Иван») или LLM по содержанию разговора (Forge `article_stage.resolve_speaker_names`).

### 5.6. `roughcut` — черновой монтаж (новое)

- **Когда:** `roughcut.enabled` и материал не помечен как уже смонтированный.
- **Вход:** транскрипт таймлайна (**не** вычитанный: нужны слова как сказаны), спикеры, звук для оценки энергии, камеры и их привязка к спикерам.
- **Выход:** `EditList` (`stages/roughcut/edit.json`).

**Детекторы** (каждый даёт кандидатов `Cut` с причиной и уверенностью):

| Детектор | Как работает | По умолчанию | Откуда код |
|---|---|---|---|
| Паузы | промежуток между словами ≥ `pauses.min_s` (1.0 с) **и** энергия ниже порога (дыхание и смех не режем); пауза сокращается до `pauses.keep_s` (0.4 с, поровну по краям) | вырезать | `recorder_word_gaps`, `_silence_spans`, `pause_spans_local` (WhisperSync) + `parse_silencedetect` (Forge) → `core/audio_spans.py` |
| Дубли | группы повторов одной фразы; всё, кроме выбранного дубля (по умолчанию последнего), вырезается от начала первой попытки до начала выбранной | вырезать | WhisperSync `engine/retakes.py` (`detect_retakes`, `refine_retakes`) — как есть |
| Начало и конец | всё до первого слова − `head_tail.pad_s` и после последнего + `pad_s` | вырезать | новое |
| Слова-паразиты | словарь по языку («ну», «как бы», «типа», «короче», «вот», «эээ») | **только маркеры** | новое |
| Нераспознанные звуки | энергия есть, а слов нет дольше 0.3 с — вероятно «эээ», которое Whisper выкинул | **только маркеры** | новое, на `core/audio_spans.py` |
| LLM (фаза 6) | лучший дубль по подаче, оговорки, «где на самом деле начинается разговор», паразиты по смыслу | выключено | `llm/` Forge, по образцу `cleanup/judge` в `analyze_stage.py` |

**Точность резов:**

- рез ставится на границу слова и прижимается к тишине (`_snap_to_word_gap`, `seam_snap_max_s` из WhisperSync);
- затем округляется до кадра;
- на звуке при программном рендере (`program`) — кроссфейд 10 мс (`crossfade_ms`).

**Мультикамера** (фаза 4):

- **привязка камера ↔ спикер:**
  - вручную на экране «Материал»;
  - автоматически (v2): кадры каждой камеры во время реплик каждого спикера → YuNet + Light-ASD (`vision/`) → чья камера показывает говорящего крупнее; камера, где почти всё время ≥ 2 лиц, — общий план (`face_crop.decide_layout`);
- **правила переключения (v1, детерминированные):**
  - реплика ≥ `multicam.min_shot_s` (2.5 с) → камера спикера;
  - короткая реплика — остаёмся на текущем плане;
  - частые перебросы (≥ 3 смены за 6 с) или наложения → общий план;
  - план дольше `multicam.max_shot_s` (20 с, опция) → вставка общего;
  - рез на `начало реплики − 0.2 с`, по кадру.

**Режимы применения:**

- `cut` (по умолчанию) — оставленные диапазоны идут встык. В местах резов маркеры с причиной («пауза 3.2 с», «дубль 2 из 3»).
- `markers` — ничего не режется, всё размечено маркерами. Этот путь по умолчанию сейчас у дублей WhisperSync.
- `disable` (оценить в фазе 3 на реальном импорте) — вырезанное остаётся на таймлайне отключёнными клипами: `enabled="0"` в FCPXML, `<enabled>FALSE</enabled>` в XML Premiere.

**Просмотр решений в интерфейсе** (фаза 3): список резов с причиной и кнопкой «прослушать» (QtMultimedia, диапазон ±2 с). Галочка «принять» меняет `Cut.accepted`. Пересчитываются только `export` и `program` (их отпечаток зависит от `edit.json`), сам `roughcut` заново не идёт.

### 5.7. `export` — проекты для монтажных программ

- **Когда:** всегда, если есть хоть одна камера.
- **Вход:** `Timeline` (размещение, синхронизированный звук, дорожки) + `EditList` (если есть) + маркеры.
- **Как:**
  1. `stages/export/sequence.py` (новое) — **нейтральная последовательность**: дорожки → клипы `(asset, src_in, src_out, rec_in, rec_out, enabled, lane, role)` + маркеры + мультикам-группы. Всё время уже в кадрах последовательности. Строится из таймлайна и `EditList`; без монтажа — один `KeepRange` на весь таймлайн.
  2. `fcpxml.py` — из WhisperSync `engine/export.py`. **Адаптировать:** вход — `Sequence` вместо `SyncPlan`. Сохраняются: рациональные времена и привязка к кадрам (`to_rational`, `_frame_rational`, `fps_to_frame_duration`), относительные пути (`_media_src`), форматы по `(fps, w, h)`, роли, политика «звук камеры выключен, если есть замена» (`source_audio_enabled`), проверки `check_fcpxml`/`validate_fcpxml`/`fcpxml_intervals`. **Новое:** мультикамера через `<media><multicam>` + `<mc-clip>` (раздел 11.4), маркеры резов.
  3. `xmeml.py` (новое) — XML Final Cut 7 (`<xmeml version="4">`) для Premiere; DaVinci Resolve его тоже читает. Premiere **не импортирует** FCPXML. Подводные камни:
     - время в кадрах, `<rate><timebase>/<ntsc>` (23.976 и 29.97 → `ntsc TRUE` с базой 24/30);
     - абсолютные `pathurl` вида `file://localhost/…` с процентным кодированием; на Windows — `file://localhost/C%3a/…`;
     - связь видео и звука (`<link>`), `sourcetrack` для каждого канала звука;
     - мультикамеру в v1 делаем «стопкой»: каждая камера на своей видеодорожке, неактивные куски `enabled = FALSE`;
     - подсэмпловая точность синхронизации в этом формате невозможна, поэтому синхронизация «зашита» в отрендеренный WAV, а позиции в кадрах.
  4. Транскрипты (`json`/`srt`) таймлайна и монтажа, мастер-WAV (`timestretch.mix_clips_on_timeline`), отчёт.
- **Проверка:**
  - автоматическая — круговое преобразование: разбор написанного XML обратно в интервалы, они должны совпадать с `Sequence` (`fcpxml_intervals` уже есть, для xmeml пишется аналог);
  - ручная — чек-лист импорта (раздел 8.4).

### 5.8. `program` — черновой мастер (новое)

- **Когда:** нужны рилсы (они режутся из мастера) или пользователь включил «отрендерить черновик».
- **Как:** ffmpeg по `EditList`. Для каждого `KeepRange` берётся видео выбранной камеры (вход файла через `Placement`) и синхронизированный голос (или рекордер со сдвигом). Куски склеиваются с кроссфейдом звука 10 мс, NVDEC → NVENC с высоким битрейтом, выбор ffmpeg — `core/ffmpeg.py` (Forge). Рядом пишется `program.transcript.json` — транскрипт таймлайна, пересчитанный во время монтажа, без Whisper.
- **Зачем:**
  - рилсы и тексты работают на одном файле — так Forge устроен сейчас, и его анализ и нарезка переносятся без переделки;
  - пользователь может посмотреть черновик без монтажной программы.
- **GPU:** NVENC.

### 5.9. `text` — вычитка и статья

- **proofread:** Forge `stages/proofread_stage.py` → `stages/text/proofread.py`, **как есть** (guardrail `is_correction_safe`, батчи, глоссарий). Вход — транскрипт таймлайна. Тайминги переносятся на исправленный текст через `core/transcript_align.py` (Forge `utils/word_alignment.py`). Результат идёт в `reels` и `article`, а `roughcut` его **не** использует.
- **article:** Forge `stages/article_stage.py` → `stages/text/article.py`, **как есть** (проверки `check_faithfulness`, разделы, имена спикеров). Вход — вычитанный транскрипт **монтажа**, чтобы дубли не попали в статью дважды.
- **term_check:** Forge `analysis/term_check.py` → `stages/text/term_check.py`.
- **Промпты** — `resources/prompts/{ru,en}/` без изменений.
- **GPU:** LLM (общая сессия).

### 5.10. `reels` — рилсы

Два подэтапа, потому что между ними должен успеть выключиться llama-server.

- **`reels-select`** (LLM). Forge `stages/analyze_stage.py` + весь `analysis/` → `stages/reels/select.py` и `stages/reels/analysis/`, **как есть**. Вход:
  - вычитанный транскрипт мастера;
  - `diarization.json` (из `speakers`, пересчитанный во время монтажа);
  - метаданные эпизода, если вход из YouTube.

  Обёртку `scripts/analyze.py` заменяет `run(ctx)`. Выход — `moments.json`, `reels_summary.md`.
- **`reels-render`** (TORCH + NVENC + WHISPER). Forge `scripts/video_processor.py` → `stages/reels/render.py`. Вход — `program.mp4`. Вместе с ним переезжают:
  - `utils/burned_subtitles.py`, `subtitle_layout.py`, `subtitle_presets.py`, `subtitle_sync.py` → `stages/reels/subtitles/`;
  - `utils/clip_intervals.py`, `reel_markdown.py` → `stages/reels/`;
  - `utils/face_*`, `active_speaker.py` → `vision/`;
  - перераспознавание каждого клипа для субтитров (`subtitle_sync`) — через общий `WhisperEngine`.
- **Позже (фаза 7):** рилсы прямо из исходников. Для вертикального кадра берётся ракурс камеры спикера, а не кроп мастера.

---

## 6. Карта переноса: что, откуда, куда, как

Как переносится файл:

- **как есть** — `git mv` + правка импортов; логика не меняется;
- **адаптировать** — меняется интерфейс: вход/выход, настройки вместо словарей, `run(ctx)` вместо `main(argv)`;
- **разрезать** — файл распадается на несколько модулей;
- **заменить** — пишется заново, старый код служит образцом;
- **не переносить** — остаётся в истории или в `contrib/`.

Все переносы делаются через `git mv`, чтобы `git log --follow` вёл историю. Пометка «как есть» ниже означает только сохранение алгоритма как кандидата после characterization-тестов; импорт без адаптации импортов, путей, ресурсов, конфигурации и обработки ошибок не считается выполненной задачей.

### 6.1. WhisperSync

| Откуда (`whispersync/…`) | Куда (`studio/…`) | Как | Заметки |
|---|---|---|---|
| `models.py` | `core/transcript.py` (Word, Segment, Transcript); `stages/sync/models.py` (Anchor, AlignmentMap); `stages/roughcut/models.py` (Take, RetakeGroup); `core/timeline.py` (MediaClip, SyncPlan → Clip, Timeline) | разрезать | `SyncResult` заменяется `StageOutput` |
| `config.py` | `core/settings/` (механизм проверки) + `SyncSettings`, `TranscribeSettings` | разрезать | поля без изменений, только разложить по разделам; `load_config` → загрузчик YAML-слоёв |
| `engine/acoustic.py` | `stages/sync/acoustic.py` | как есть | |
| `engine/matcher.py` | `stages/sync/matcher.py` | как есть | `WhisperSyncConfig` → `SyncSettings` |
| `engine/timestretch.py` | `stages/sync/timestretch.py` | как есть | `mix_clips_on_timeline` используется и в `export` |
| `engine/strategies.py` | `stages/sync/strategies.py` | как есть | + соответствие режимов (`simple` → 1, `complex` → 3) |
| `engine/pipeline.py` | `scan`, `transcribe`, `sync/*`, `roughcut`, `export` | разрезать | по таблице в 5.3. `run_pipeline` и `_run_pipeline_locked` заменяются `stages/sync/stage.py` (обвязка заново, функции как есть) |
| `engine/transcriber.py` | `stages/transcribe/engine.py` | адаптировать | основа единого движка; + функции из Forge `transcribe_stage.py` |
| `engine/transcript_export.py` | `core/transcript.py` | как есть | единый (де)сериализатор формата |
| `engine/media.py` | `core/media.py` | как есть | |
| `engine/proc.py` | `core/proc.py` | как есть | |
| `engine/naming.py` | `core/naming.py` | как есть | используется сканером |
| `engine/sources.py` | `core/sources.py` | как есть | id ассетов |
| `engine/workspace.py` | `core/workspace.py` | как есть | + семантика Forge `run_lock` для `batch` |
| `engine/retakes.py` | `stages/roughcut/retakes.py` | как есть | |
| `engine/self_check.py` | `stages/sync/self_check.py` | как есть | |
| `engine/verify.py` | `stages/sync/verify.py` | как есть | |
| `engine/separation.py` | `stages/sync/ambience.py` + `modules/manager.py` | разрезать | поиск `.sep-venv` (`sep_venv_candidates`, `separator_python`) обобщается в менеджер модулей |
| `engine/enhance.py` | `stages/sync/enhance.py` + `modules/` | разрезать | то же |
| `engine/system_check.py` | `core/capabilities.py` + `cli doctor` | адаптировать | слить с Forge `preflight.py` |
| `logging_setup.py` | `core/logging.py` | адаптировать | слить с Forge `utils/logging_utils.py` |
| `cli.py` | `cli/main.py` | заменить | флаги → `--set`; старые флаги — псевдонимы на одну версию |
| `app.py` | `app.py` | адаптировать | |
| `gui/main_window.py` | `gui/main_window.py` + `gui/screens/*` | разрезать | переиспользовать: drag&drop, восстановление состояния, `_install_quiet_message_handler`, `load_stylesheet`, закрытие с отменой |
| `gui/worker.py` | `gui/bridge.py` | заменить | QThread с конвейером в процессе → QProcess + события; веса прогресса (`overall_progress`) — как есть |
| `gui/widgets/drop_zone.py`, `log_view.py`, `timeline_preview.py`, `strategy_diagram.py`, `sync_simulator.py`, `help_page.py` | `gui/widgets/` | как есть | `timeline_preview` дорисовывает резы и смены камер (фаза 3–4) |
| `gui/widgets/settings_dialog.py` | `gui/screens/settings.py` | адаптировать | становится экраном всех настроек |
| `gui/theme.qss` | `gui/theme.qss` | как есть | |
| `tools/verify_sync.py` | `cli verify` | адаптировать | |
| `whispersync.spec` | `packaging/pyinstaller/studio.spec` | адаптировать | |
| `setup_venv.sh`, `setup_sep_venv.sh`, `requirements-sep.txt` | `modules/recipes/` | адаптировать | |
| `tests/*` (403 теста в проверенном дереве) | `tests/core`, `tests/sync`, `tests/roughcut`, `tests/export`, `tests/gui` | адаптировать через фасады | тесты GUI-воркера переписываются под `bridge`; число фиксируется после snapshot |
| `README*.md` (разделы «Sync Strategies», «Retake», «Self-Check», «Voice Enhancement», «Troubleshooting») | `docs/sync.md`, `docs/troubleshooting.md` | адаптировать | |
| `PROJECT_ANALYSIS.md`, `PROJECT_AUDIT_2026-09-08.md`, `TODO_project.md`, `CHANGELOG.md` | `docs/history/whispersync/` | как есть | архив |
| `docs/images/*` | `docs/images/` | как есть | скриншоты обновить после фазы 1 |

### 6.2. Podcast Reels Forge

| Откуда (`podcast_reels_forge/…`) | Куда (`studio/…`) | Как | Заметки |
|---|---|---|---|
| `pipeline.py` | разные места, см. ниже | разрезать | |
| ↳ `PipelineIO`, `pick_input_file`, `find_input_queue`, `_ensure_audio_companions`, `_ensure_mp3_companion` | `stages/scan`, `core/media.py` | адаптировать | ввод одного файла — частный случай скана |
| ↳ `_LlamaSession`, `_kill_llama_server`, `_autotune_llama_cpp_conf`, `_merge_llama_cpp_conf`, `_get_model_overrides`, `_prompt_variant_for_model` | `llm/session.py` | как есть | |
| ↳ `_has_cuda`, `_gpu_vram_gb`, `_cpu_count`, `_autotune_video_threads` | `core/capabilities.py` | как есть | |
| ↳ `_PipelineRun`, `EpisodeState`, `stage_*` | `stages/runner.py`, `Project`, `run()` этапов | заменить | логика каждого `stage_*` уходит в `run()` своего этапа |
| ↳ `run_module` | `stages/runner.py` | адаптировать | + чтение событий JSON-lines |
| ↳ `PIPELINE_STAGES`, `resolve_stages` | `stages/planner.py` | заменить | `--only`/`--skip` сохраняются |
| ↳ `_youtube_conf`, `_run_youtube_fetch` | `stages/fetch` | адаптировать | |
| `config.py`, `utils/config_loader.py` | `core/settings/` | заменить | словари → dataclass'ы с проверкой; оверлей `*.local.yaml` сохраняется |
| `autonomy.py` | `cli/batch.py` | адаптировать | лог-файл, отчёт, уведомления |
| `run_report.py` | `core/report.py` | как есть | |
| `preflight.py` | `core/capabilities.py` | адаптировать | слить с `system_check` |
| `cli.py`, `__main__.py`, корневые `start_forge.py`, `transcribe_input_audio.py`, `rerender_videos.py` | `cli/main.py` | заменить | перезапуск в venv из `start_forge.py` → лаунчер; `rerender` → `studio run --only reels-render` (отпечатки делают остальное) |
| `stages/transcribe_stage.py` | `stages/transcribe/` | разрезать | см. 5.2; CLI и `parse_args` — не переносить |
| `stages/proofread_stage.py` | `stages/text/proofread.py` | адаптировать | `main(argv)` → `run(ctx)` |
| `stages/article_stage.py` | `stages/text/article.py` | адаптировать | то же |
| `stages/analyze_stage.py` | `stages/reels/select.py` | адаптировать | то же; асинхронный код как есть |
| `stages/fetch_stage.py`, `scripts/fetch_youtube.py`, `sources/*` | `stages/fetch/` | адаптировать | |
| `stages/video_stage.py` | `stages/reels/` | как есть | |
| `analysis/speaker_turns.py` | `stages/speakers/turns.py` | как есть | общий для text/reels/roughcut |
| `analysis/transcript_index.py` | `core/transcript_index.py` | как есть | нужен и roughcut |
| `analysis/audio_features.py` | `core/audio_spans.py` (разбор silencedetect/volumedetect) + `stages/reels/analysis/` (аннотация кандидатов) | разрезать | |
| `analysis/term_check.py` | `stages/text/term_check.py` | как есть | |
| остальные `analysis/*` (chunking, contracts, decisions, metadata, ranking, scoring, serializers, validation, candidate_extraction) | `stages/reels/analysis/` | как есть | |
| `llm/providers.py`, `llm/schemas.py` | `llm/` | как есть | |
| `utils/llama_cpp_service.py` | `llm/server.py` | как есть | |
| `utils/json_utils.py` | `llm/json_utils.py` | как есть | |
| `scripts/diarize.py` | `stages/speakers/pyannote.py` | адаптировать | запуск в окружении `diarization` |
| `scripts/video_processor.py` | `stages/reels/render.py` | адаптировать | вход — `program.mp4`; `ffmpeg_cut`, превью, экспорт webm/gif/аудио — как есть |
| `scripts/rerender_videos.py` | — | заменить | повторный запуск этапа |
| `scripts/evaluate_prompts.py` | `tools/evaluate_prompts.py` | как есть | инструмент разработчика |
| `scripts/host_memory.py`, `utils/host_memory.py` | `contrib/hooks/host_memory/` | адаптировать | хук `before_gpu`/`after_gpu`, а не часть ядра |
| `scripts/transcribe.py`, `scripts/analyze.py` | — | не переносить | тонкие обёртки |
| `utils/face_track.py`, `face_crop.py`, `active_speaker.py` | `vision/` | как есть | используются и `reels`, и (позже) `roughcut` |
| `utils/burned_subtitles.py`, `subtitle_layout.py`, `subtitle_presets.py`, `subtitle_sync.py` | `stages/reels/subtitles/` | как есть / адаптировать | `subtitle_sync.load_model` → общий `WhisperEngine` |
| `utils/word_alignment.py` | `core/transcript_align.py` | как есть | |
| `utils/clip_intervals.py` | `stages/reels/clip_intervals.py` | как есть | |
| `utils/reel_markdown.py` | `stages/reels/markdown.py` | как есть | |
| `utils/media_qa.py` | `core/media_qa.py` | как есть | |
| `utils/ffmpeg.py` | `core/ffmpeg.py` | как есть | |
| `utils/fingerprint.py` | `core/fingerprint.py` | как есть | `StageState` — основа отпечатков исполнителя |
| `utils/env.py` | `core/env.py` | как есть | |
| `utils/run_lock.py` | `core/workspace.py` | заменить | на `output_lock` WhisperSync (pid, перехват устаревшей блокировки) |
| `utils/logging_utils.py` | `core/logging.py` | заменить | |
| `prompts/{ru,en}/*` | `resources/prompts/{ru,en}/` | как есть | |
| `assets/fonts/*` (+ licenses), `assets/models/*` | `resources/fonts/`, `resources/models/` | как есть | модели — кандидаты на скачивание при первом запуске, а не в репозитории |
| `assets/subtitles/style-editor.html`, `gui/assets/editor.js`, `editor.css`, `subtitle-presets.js`, `gui/subtitles.html` | `gui/`, `resources/` | перенести поведение и пресеты | нативный редактор внутри PyQt GUI; читает и пишет раздел `reels.subtitles` и правки субтитров через приложение |
| `gui/*.html`, `gui/assets/app.js`, `app.css` (сборщик `config.yaml`) | — | не переносить | заменяется экраном настроек в Qt |
| `tests/*` (620 тестов в проверенном дереве) | `tests/text`, `tests/reels`, `tests/speakers`, `tests/fetch`, `tests/core`, `tests/llm` | адаптировать через фасады | `test_pipeline.py`, `test_pipeline_efficiency.py`, `test_unattended_runs.py`, `test_stage_decisions.py` переписываются под планировщик и исполнитель |
| `docs/USER_GUIDE.md`, `CONFIGURATION.md`, `AUTONOMOUS.md`, `PROMPTS.md`, `DEVELOPMENT.md` | `docs/` | адаптировать | конфигурация — под новые разделы, AUTONOMOUS → `batch` |
| `docs/COMPETITOR_REVIEW.md`, `ANALYSIS_REPORT.md`, `AUTONOMY_REVIEW.md`, `CHANGELOG.md` | `docs/history/forge/` | как есть | архив |
| `local/*` (не в git) | остаётся на хосте | адаптировать | `forge-local.sh`, `forge-night.sh` вызывают `studio run`/`studio batch`; `pedobraz_pause.py` подключается хуком |

### 6.3. Соответствие настроек (для `migrate_settings.py`)

| Было | Стало |
|---|---|
| WhisperSync: `model`, `device`, `compute_type`, `language`, `vad_filter`, `beam_size`, `batch_size`, `best_of`, `patience`, `condition_on_previous_text`, `repetition_penalty`, `no_repeat_ngram_size`, `transcribe_mode`, `quality_beam_size`, `initial_prompt`, `use_cache`, `cache_dir`, `cache_max_age_days` | `transcribe.*` |
| WhisperSync: `default_strategy`, `recorder_mode`, `audio_source_camera`, `timebase_source`, `camera_av_offset_ms*`, `crossfade_*`, `output_audio_format`, `stretch_method`, `seam_snap_max_s`, `render_workers`, `min_anchors`, `anchor_*`, `phrase_gap_threshold`, `match_window_margin`, `seed_*`, `acoustic_*`, `gcc_eps`, `alignment_*`, `boundary_flex`, `flex_*`, `pause_duck_*`, `ambience_*`, `voice_enhance`, `self_check_*`, `voice_segment_minutes`, `reuse_source_dir` | `sync.*` |
| WhisperSync: `detect_retakes`, `retake_*` | `roughcut.retakes.*` |
| WhisperSync: `fcpxml_version`, `save_transcripts`, `render_master_wav`, `output_dir` | `export.*` |
| WhisperSync: `video_exts`, `audio_exts`, `probe_timeout_s` | `scan.*` |
| Forge: `transcription` | `transcribe.*` |
| Forge: `diarization` | `speakers.*` |
| Forge: `proofread`, `article` | `text.*` |
| Forge: `processing`, `subtitles`, `video`, `audio`, `exports`, `prompts` | `reels.*` |
| Forge: `llama_cpp` | `llm.*` |
| Forge: `autonomy`, `cli` | `batch.*` |
| Forge: `youtube` | `fetch.*` |
| Forge: `cache` | `transcribe.cache` + `llm.cache` |
| Forge: `host_memory` | `contrib/hooks/host_memory` + `hooks.*` |
| Forge: `paths` | рабочая папка проекта + `batch.*` |

### 6.4. Что пишется с нуля

| Компонент | Объём | Фаза |
|---|---|---|
| `core/project.py`, `core/timeline.py` (модель, четыре шкалы, `AudioWarpMap`, круговые тесты) | средний | 1–2 |
| `stages/base.py`, `planner.py`, `runner.py`, `worker.py`, `events.py` | средний | 1 |
| `stages/scan` (классификация, группировка, главы, дорожки рекордера) | средний | 1–2 |
| `gui/screens/*`, `gui/bridge.py` | крупный | 1–3 |
| `cli/main.py` | малый | 1 |
| Режимы `camera`/`simple`/`auto`, экспорт без рендера при `k ≈ 1` | средний | 2 |
| `stages/timeline` | малый | 2 |
| `export/sequence.py`, `export/xmeml.py` | средний | 2 |
| `stages/roughcut` (детекторы, `EditList`, просмотр решений) | крупный | 3 |
| `stages/program` | малый–средний | 3 |
| `speakers/mics.py`, мультикамера (правила, привязка, `mc-clip`) | средний | 4 |
| `modules/manager.py`, установщики, мастер первого запуска | средний | 6 |
| LLM-детекторы чернового монтажа | средний | 6 |

---

## 7. Фазы работ

После каждой фазы есть работающее приложение. Фаза закрыта, когда выполнены все критерии готовности.

### Актуальный срез — 2026-10-10

Галочки ниже относятся к выполнению конкретной задачи, а не к приёмке всей фазы.
Обозначения: `[x]` — реализация выполнена в указанном объёме; `[ ] Частично` —
есть реализация, но перечисленный остаток не закрыт; `[ ] Не выполнено` — работы
ещё нет; `[ ] Приёмка` — требуется проверка на модели/реальном материале/внешнем ПО.
Любая частичная задача содержит явное описание недоделанного.
Незакрытая задача может быть частично реализована: оставшаяся часть указана в таблице.
Подробный актуальный backlog и доказательства — в начале `docs/IMPLEMENTATION_STATUS.md`;
его последующий журнал хранит исторические, а не текущие списки пробелов.

| Фаза | Текущее состояние | Что мешает закрытию |
|---|---|---|
| 0 | Истории, snapshots/patch, фикстуры и локальные сборки есть | CI/чистая установка, реальная запись, квалификация Python 3.12 |
| 1 | Каркас и локальный pipeline реализованы | Полная golden-регрессия и сквозная GUI/CLI приёмка |
| 2 | Режимы sync, voice/warp, FCPXML/XMEML, ручные роли, RecorderTrack и drag&drop групп есть | Паритет выбора потоков, раздельные сессии, реальный импорт NLE |
| 3 | Паузы/дубли, решения, обзор резов на таймлайне, прослушивание без мастера, CFR-мастер с кроссфейдами есть | Реальное прослушивание, проверка слов на стыках, сквозной переэкспорт |
| 4 | Mics/pyannote facade, выбор камер (GUI/CLI), политика общего плана, дорожки спикеров и multicam есть | Нелинейная mic-карта, модели и NLE приёмка |
| 5 | Тексты/term-check, role routing, рилсы/аудиокэш, captions/editor, vision, fetch, channel/batch с уведомлениями и отменой загрузок, миграция настроек есть | Raw warp replay, разделы для непереносимых настроек, phase-wide reuse и Forge-регрессия |
| 6 | Модули, загрузка YuNet/Light-ASD/Whisper/GGUF, LLM-детекторы монтажа, мастер настройки и Linux frozen есть | Приёмка LLM-монтажа на модели, измеряемое tuning, production installers |
| 7 | Последующая дорожная карта | Не блокирует первоначальный локальный выпуск |

**Ни одна фаза ещё не закрыта по всем критериям приёмки.** Текущая база —
447 passed, 2 skipped на Python 3.10.20; Linux installed/frozen smoke проходит, включая
CLI/worker/resources и offscreen GUI startup. Это не все исходные 403 + 620 тестов,
не реальный ML-прогон, не NLE-приёмка и не чистая установка. Python 3.12 — целевая
release-база, её квалификация в этом окружении ещё нужна. Linux приоритетен;
остальные ОС следуют после сквозной локальной проверки по решению владельца.

### Фаза 0. Подготовка и фиксация исходного состояния

**Задачи:**

- [x] Принять решения из раздела 2. См. `docs/PROJECT_DECISIONS.md`.
- [x] Определить политику исходных репозиториев: по решению владельца не изменяем их; разработка только в Ultimate Video Forge.
- [x] Snapshot manifest фиксирует HEAD/окружения/состояние исходников; WhisperSync patch сохранён с SHA256, отдельные lock-файлы присутствуют. Исторический импорт описан в `docs/source-snapshots/README.md`; исходники теперь не меняем.
- [x] Создать репозиторий и импортировать оба проекта с историей: реализация Studio импортирована merge с сохранением обоих subtree и исходной истории Ultimate Video Forge.
  ```bash
  git init studio && cd studio && git commit --allow-empty -m "chore: start"
  git subtree add --prefix=legacy/whispersync /home/borm/VibeCoding/WhisperSync main
  git subtree add --prefix=legacy/forge "/srv/storage/docs/Podcast Reels Forge" main
  ```
- [ ] **Частично:** CI definitions и отдельные locks есть. Не доделано: подтверждённый запуск обоих legacy suites и smoke-матрицы 3.10–3.13; Python 3.14 не квалифицирован.
- [ ] **Частично:** wheel/frozen build и same-host installed resources/CLI/GUI smoke выполнены. Не доделано: подтверждённый CI и чистое окружение, реальные поздние ML-импорты.
- [x] Скрипт `tools/make_fixtures.py`: детерминированный синтетический проект с двумя камерами, рекордером, речью, дрейфом, сдвигом, главой GoPro, паузой и повтором фразы. Локально генерируется; сквозной ML-прогон в CI ещё нужен.
- [ ] **Приёмка:** выбрать приватную запись 20–40 минут (2 камеры + рекордер); в workspace пока не выбрана.

**Готово, когда:** исходные тесты зелёные (текущая фактическая база 403 + 620), patchset WhisperSync учтён, фикстуры генерируются одной командой, а wheel/PyInstaller smoke-тесты проходят на чистом окружении.

### Фаза 1. Контракты и каркас: «отдал папку — получил диагностированный проект»

Проверенные подзадачи миграции:
- [x] Fingerprint транскрипции включает SHA-256 выбранных исходников, даже при сохранении размера и mtime.

**Задачи:**

- [ ] **Частично:** characterization для transcript/export, fingerprints, locks/processes и PCM parity есть. Не доделано: полная golden-матрица исходных workflow и сравнение на реальной записи.
- [x] `core/`: `Project`, `SourcePlacement`, `AudioWarpMap`, `EditMap`, schema-version и проверяемые manifests реализованы; полный legacy-паритет проверяется отдельно.
- [x] Stage API, discover/planner/runner/worker/events: повторное планирование, приватные worker-результаты, проверка артефактов, публикация с rollback journal и recovery. Это восстанавливаемая многофайловая публикация, а не одновременная видимость всех файлов.
- [x] `scan` v1: подпапки, шаблоны имён, теги, главы, роли и сохранение ручных overrides.
- [x] `transcribe`: Whisper facade, выбор входов, явный поток, content cache, OOM-лестница. Реальный модельный прогон остаётся приёмкой.
- [x] `prepare` + `transcribe`: нужные входы объявляются планировщиком, effective settings сохраняются для worker.
- [ ] **Частично:** `sync complex` planner/renderer, voice WAV, warp maps и PCM parity реализованы. Не доделано: полный characterization и реальная WhisperSync-регрессия.
- [x] Нейтральный `Sequence` и FCPXML/XMEML адаптеры реализованы; экспорт синхронизированного звука, rates/channels/roles покрыт локальными регрессиями. NLE-приёмка отдельно.
- [ ] **Частично:** Qt-экраны/bridge, запуск/лог/отмена/Material и Settings dirty guards есть. Не доделано: drag/drop, полный timeline preview и визуальная/клавиатурная приёмка.
- [x] `cli`: `scan`, `plan`, `run`, `doctor` (публичный entry point `uvf`).
- [ ] **Частично:** перенесены выбранные регрессии. Не доделано: весь baseline suite и golden parity; legacy сохраняется.

**Готово, когда:**

- на фикстурах и на реальной записи интервалы FCPXML (`fcpxml_intervals`) совпадают с выдачей старого WhisperSync на тех же входах при тех же настройках;
- все перенесённые тесты WhisperSync зелёные;
- запуск из интерфейса и из `studio run <папка>` даёт одинаковый результат;
- повторный запуск без изменений ничего не пересчитывает только после проверки manifest; повреждённый/неполный артефакт автоматически пересоздаётся.

### Фаза 2. Контракт времени, режимы синхронизации и Premiere

**Задачи:**

- [x] `AudioWarpMap` и preserved strategy 1/2/3 planner/renderer реализованы; приближённый k не заменяет кусочную модель. Реальный полный parity — отдельная приёмка.
- [ ] **Частично:** camera/simple/auto, acoustic fallback, reference-camera и независимые пары recorder/camera реализованы. Раздельные сессии связываются общими записями либо явными `sync.session_offsets`; неподтверждённые смещения не угадываются. Выбранный камерный поток экспортируется отдельным PCM для NLE. Не доделано: полный legacy workflow parity и реальная NLE-приёмка.
- [x] Camera-only acoustic placement реализован и проверен generated-audio regression.
- [x] Timeline transcript строится через placement без повторного Whisper; синхронизированный voice имеет отдельные warp/rendered contracts.
- [ ] **Частично:** transcript/SRT, confidence, initial_prompt, glossary/hotwords и anti-repetition controls есть; текст сегмента сохраняется без word alignment. Не доделано: подтверждённый полный sentence/glossary/hotwords parity Forge на реальных моделях.
- [x] Sequence → FCPXML/XMEML, структурные/interval regression tests реализованы. Реальный NLE импорт не закрыт этой отметкой.
- [x] Роли/группы/device редактируются и переживают rescan; группы меняются drag&drop на экране «Материал» (несколько файлов сразу). Реальная GUI-приёмка отдельно.
- [x] `RecorderTrack` в модели проекта (каналы, спикер, имя, включена), предложение дорожек при scan с сохранением ручных, GUI/CLI (`uvf tracks`) объединения каналов, имён спикеров и отключения; `speakers` берёт именованные дорожки. Реальная GUI-приёмка отдельно.
- [x] Typed warp validation проверяет порядок/перекрытия/покрытие согласно контракту; tests присутствуют.

**Готово, когда:**

- один и тот же проект открывается в Final Cut и Premiere по чек-листу 8.4, синхронизация совпадает на глаз и на слух в начале, середине и конце;
- `auto` выбирает `simple` на короткой записи с общими часами и `complex` на длинной с дрейфом (на фикстурах — автоматически, на реальной записи — вручную);
- WhisperSync можно архивировать: всё, что он умел, есть в новом приложении.

### Фаза 3. Черновой монтаж v1

**Задачи:**

- [x] Общие word-gap/silence span utilities и energy-gated roughcut реализованы.
- [ ] **Частично:** pauses/head/tail/retakes, filler markers (включая «как бы»), маркеры unknown-sound (не тишина и не слово), общая защита границ резов, пресеты gentle/balanced/tight (`uvf preset`, Settings) и `tools/check_program.py` (перераспознавание мастера) есть. Не доделано: реальные preset/listening проверки и прогон проверки слов на реальной модели.
- [ ] **Приёмка:** cut/markers и XML reasons реализованы; не проверены disable/видимость/семантика в реальном NLE.
- [x] `program`: placement-aware мастер с выбранными камерами и синхронизированным звуком, CFR и отображением транскрипта на фактические кадровые интервалы; кроссфейды звука на стыках (`program.crossfade_ms`, по умолчанию 10 мс) без изменения длительности. Приёмка стыков на слух/длинного A/V отдельно.
- [ ] **Частично:** accept/reject и прослушивание «с резом / оригинал» реализованы, в том числе без мастера (синхронизированный голос камеры или исходник транскрипта через placement). Не доделано: реальное прослушивание и сквозная проверка переэкспорта.
- [x] Обзор резов на всём таймлайне (оставленное, принятые и предложенные резы, маркеры, playhead, длина программы) синхронизирован со списком решений. Видео-предпросмотр стыка не входит; визуальная приёмка отдельно.
- [ ] **Приёмка:** keep_fillers/initial_prompt на реальной записи не проверены.

**Готово, когда:**

- ни один рез не попадает внутрь слова. Проверка автоматическая: мастер перераспознаётся, и слова сравниваются с транскриптом монтажа (идея и код самопроверки `self_check.diagnose_words`);
- на реальной записи час материала сокращается без «рваных» мест — оценка на слух по чек-листу;
- пороги пауз вынесены в настройки и пресеты.

### Фаза 4. Спикеры и мультикамера

**Задачи:**

- [ ] **Частично:** mics/smoothing/pyannote facade есть. Не доделано: turns/manual naming parity, cross-file identity, nonlinear mic attribution и real model acceptance.
- [x] speaker-camera mapping задаётся на экране «Материал» и `uvf cameras` (хранится в проекте, перекрывает `program.speaker_cameras`); общий план назначается так же (`@wide`, «Wide shot»).
- [ ] **Приёмка:** правила 5.6 реализованы: camera_id, minimum shot, coverage fallback, manual precedence, наложения и частые перебросы → общий план, вставка общего в длинный план (`max_shot_s`), упреждение реза `cut_lead_s`. Не проверена реальная оценка переключений.
- [ ] **Приёмка:** retimed multicam/voice/ambience/speaker selection и XMEML lanes реализованы, generated FCPXML прошёл 1.9 DTD. Не проверены импорт/переключение/playback в NLE.
- [x] `program` учитывает выбранную камеру, ручные решения и speaker-camera mapping с проверкой покрытия.

**Готово, когда:** запись «2 камеры + 2 петлички» даёт в Final Cut мультикам с разумными переключениями, и ракурс на любом куске меняется штатно (клавишами или в инспекторе).

### Фаза 5. Тексты, рилсы и совместимость Forge

**Задачи:**

- [ ] **Частично:** provider/cache/session/retries/role routing есть; `tools/migrate_settings.py` переносит старые настройки по таблице 6.3 с отчётом по каждому ключу. Не доделано: разделы Studio для ключей, которые пока попадают в отчёт как unmapped (batch/fetch/hooks и часть reels/llm), и phase-wide model reuse.
- [x] Вычитка/realignment, статья и term-check интеграция реализованы: ручные offline fixes, отдельный network opt-in, кэш и отчёт. Реальная приёмка отдельно.
- [ ] **Частично:** verified selection/refine/judge/ranking/context, role routing и content-based audio probes/cache реализованы. Voice/placement/edited mapping позволяют работать без постоянного мастера/program. Не доделано: raw warp replay без voice artifacts, полный legacy settings и moments golden parity.
- [ ] **Частично:** source/program reels, captions и isolated tracking/fallback есть. Не доделано: полный render parity, реальные vision/frozen inference и визуальная приёмка.
- [x] `fetch`: single URL в новую/пустую папку с metadata. Channel discovery/acquisition также реализован; реальная загрузка и night-run — отдельная приёмка.
- [ ] **Частично:** project/channel queue, atomic reports, queue_finished event, resume, night helper, внешние уведомления (webhook/desktop) и отмена посреди загрузки с докачкой реализованы. Не доделано: phase-wide reuse и overnight acceptance. Shell hooks отложены по PROJECT_DECISIONS.md.
- [x] Нативный редактор субтитров внутри PyQt GUI с видео/overlay, правкой текста/таймингов/стилей и публикацией JSON/SRT/ASS. Визуальная приёмка отдельно.
- [ ] **Частично:** выбранные Forge regressions перенесены; `tools/legacy_suites.py` гоняет frozen-наборы как сравнительные (по-тестовые отчёты, сравнение со снимком 403/620 и предыдущим прогоном). Не доделано: golden workflow parity Studio против этих наборов; legacy не удаляем до подтверждения.

**Готово, когда:**

- все 620 тестов Forge из зафиксированного snapshot зелёные плюс characterization-тесты порядка, кэшей и аварийного завершения;
- на одном и том же `program.mp4` и с тем же кэшем LLM `moments.json` совпадает с выдачей старого Forge, а рилсы по содержанию те же;
- ночной прогон канала (`batch`) отработал хотя бы одну ночь без вмешательства и не остановил чужой llama-server;
- Forge можно архивировать.

### Фаза 6. LLM в монтаже, модули, установщики

**Задачи:**

- [x] LLM-детекторы монтажа (подача дубля, оговорки, начало разговора, смысловые паразиты) с JSON/evidence guardrails: только номера фраз и дословные цитаты, времена из слов транскрипта, отклонённое — в `llm-report.json`, по умолчанию предложения. Приёмка на реальной модели отдельно.
- [ ] **Частично:** modules manager/recipes/GUI/progress/cancel/locks есть. Не доделано: полнота stage → install navigation и реальные ML установки/запуски.
- [x] YuNet/Light-ASD, Whisper (base…large-v3-turbo) и Gemma 4 GGUF загружаются из приложения: закреплённые ревизии, проверка хэшей, докачка, отмена, подстановка в настройки; мастер первого запуска показывает скачанное, предлагает large-v3-turbo и сохраняет выбранные Whisper/GGUF в проект. Реальная загрузка на чистой машине — приёмка фазы.
- [ ] **Частично:** Linux frozen, setup wizard и local install script (binary/link/desktop entry) есть; same-host installed smoke пройден. Не доделано: dependency bundling, clean-machine/Python 3.12 и Windows/macOS; бинарник требует rebuild после последних guards.
- [ ] **Частично:** compute probe/recommendations есть. Не доделано: измеряемое memory/performance autotuning Whisper/llama.

**Готово, когда:** на чистой машине установщик → первый запуск → проект из фикстур проходит без ручной установки чего-либо, кроме драйвера NVIDIA.

### Фаза 7. Дальше (по приоритету)

- [ ] **Частично:** source-video reels есть; не доделан speaker-angle выбор для рилсов, per-reel framing и instant crop preview.
- [ ] **Не выполнено:** vision авто-привязка камер к спикерам.
- [ ] **Не выполнено:** OTIO/DaVinci Resolve экспорт.
- [ ] **Не выполнено:** MLX/VideoToolbox на Mac.
- [ ] **Частично:** нативный subtitle editor есть; последующие расширения/визуальная приёмка остаются.
- [ ] **Не выполнено:** −14 LUFS, обложки, отдельный hook-title layout, аудиограмма и YouTube most-replayed integration.

---

## 8. Тестирование и приёмка

### 8.1. Модульные тесты

- 403 теста WhisperSync и 620 тестов Forge — текущая проверенная база; после фиксации исходных commit/patchset число тестов фиксируется в CI. Тесты оркестраторов (Forge `test_pipeline*.py`, `test_unattended_runs.py`, `test_stage_decisions.py`; WhisperSync `test_pipeline.py`, `test_worker_progress.py`, `test_gui_cancel.py`) переписываются под планировщик, исполнитель и мост.
- Новые обязательные тесты:
  - преобразования четырёх шкал и `AudioWarpMap` (`core/timeline`): покрытие, монотонность, кусочный Hybrid, VFR, отрицательная калибровка;
  - решения планировщика: таблица «содержимое папки + настройки → план» на десятке сценариев (одна камера без рекордера, две камеры + рекордер, только звук, YouTube, нет CUDA, нет модуля);
  - круговые тесты FCPXML и xmeml (`Sequence` → XML → интервалы);
  - детекторы чернового монтажа на синтетических транскриптах;
  - `speakers.mics` на синтетических дорожках с перекрёстной слышимостью.

### 8.2. Сквозные тесты на фикстурах

`tools/make_fixtures.py` → `studio run <фикстура> --json` в CI (без GPU: `compute.allow_cpu=true`, модель Whisper `tiny`). Проверяется:

- найденные устройства;
- оценка дрейфа в пределах допуска;
- интервалы в XML;
- резы пауз и дублей на ожидаемых местах.
- повторный запуск после убийства процесса сохраняет предыдущий успешный результат;
- неполный manifest не приводит к ложному `reuse`;
- worker не захватывает чужой lock или llama-server.

### 8.3. Регрессия против старых версий

- **Фаза 1–2:** старый WhisperSync и новое приложение на одних и тех же входах дают одинаковые интервалы FCPXML и одинаковые WAV (сравнение по хэшу или с допуском по сэмплам).
- **Фаза 5:** старый Forge и новое приложение на одном `program.mp4` с общим кэшем LLM дают одинаковый `moments.json`.

Сравнение выполняется с зафиксированными commit SHA, настройками, моделью и набором prompts. «Одинаково по содержанию» дополняется машинным сравнением схемы, временных диапазонов, ссылок на исходное аудио и статуса каждого кандидата; допускаются только заранее описанные поля, зависящие от недетерминированного порядка.

### 8.4. Ручная приёмка импорта (чек-лист)

Для Final Cut Pro (версия ___) и Premiere Pro (версия ___), на реальной записи:

- [ ] файл открывается без ошибок и предупреждений, все медиа найдены (нет «offline»);
- [ ] частота кадров и размер последовательности верные;
- [ ] синхронизация губ в начале, середине и конце каждого клипа;
- [ ] звук камеры выключен там, где есть синхронизированный голос, и есть там, где его нет;
- [ ] резы не внутри слов, маркеры видны и подписаны;
- [ ] роли (Final Cut) или имена дорожек (Premiere) понятны;
- [ ] мультикам: ракурс переключается штатно (Final Cut), стопка дорожек корректна (Premiere);
- [ ] после переноса папки с материалом проект находит медиа (относительные пути в FCPXML, повторное связывание в Premiere).

Для проверки нужны Mac с Final Cut и машина с Premiere. Без них фазы 2–4 нельзя считать закрытыми.

---

## 9. Риски

| Риск | Чем грозит | Что делаем |
|---|---|---|
| Объём работ растёт без конца | приложение никогда не «готово» | каждая фаза выпускаемая; заморозка функций в старых репозиториях; фаза 7 — только после 1–6 |
| Нюансы XML для Premiere (NTSC, каналы звука, связи) | проект открывается криво | круговые тесты + ручной чек-лист на каждой фазе; начать с простых случаев (25/50 к/с, стерео) |
| Whisper выбрасывает «ээ»/«мм» | паразиты не находятся | `initial_prompt` с примерами, детектор «энергия без слов», по умолчанию только маркеры |
| Сканер путает устройства | неверная синхронизация | группировка по подпапкам в приоритете; экран «Материал» с ручной правкой до запуска; предупреждения |
| Нехватка видеопамяти (Whisper + llama + torch) | падения по OOM | этапы с GPU строго по одному в отдельных процессах; одна сессия llama на группу; хуки освобождения памяти |
| Размер установки (torch, CUDA) | неподъёмный установщик | модули, которые ставятся при первом включении (фаза 6) |
| Ночные прогоны Forge перестанут работать | регресс в привычном процессе | старый Forge работает до закрытия фазы 5; `batch` проверяется ночным прогоном |
| Расхождение шкал времени | сдвиги на секунды в экспорте и рилсах | единый модуль преобразований, круговые тесты, запрет «ручной арифметики» времени в этапах (проверка на ревью) |
| Ложный reuse после сбоя или смены файла | пользователь получает старый/неполный артефакт | manifest со статусом `ok`, атомарная публикация, checksum/identity входов и тест kill/resume |
| Цикл auto-sync и поздней транскрипции | план зависает или непредсказуемо меняется во время запуска | `prepare` фиксирует полный набор транскрипций до DAG; номер ревизии плана сохраняется |
| Чужой llama-server/опасный hook | остановка другого процесса или потеря данных | PID/process group, owner marker, allowlist, запрет `pkill -f`, тест второго процесса |
| Различия VFR, контейнерной и picture-длительности | offline/неверные границы в NLE | хранить обе длительности, экспортировать frame-accurate picture duration и проверять реальные файлы |
| Лицензии моделей и ассетов | нельзя легально собрать или распространять установщик | SBOM/NOTICE для кода, моделей, шрифтов, ffmpeg и политик скачивания до релиза |
| Нет доступа к Final Cut или Premiere для проверки | фазы не закрываются | заранее решить, на каких машинах и версиях проверяем |
| Лицензия | юридическая неясность при публикации | решить до первого публичного коммита (раздел 2) |

---

## 10. Первые шаги

1. Ответить на вопросы раздела 2 и зафиксировать commit/patchset обоих проектов, включая незакоммиченные изменения WhisperSync.
2. Фаза 0: новый репозиторий, `git subtree`, CI, wheel/PyInstaller smoke, фикстуры и golden-артефакты.
3. Отдельно согласовать контракты `project.json`, manifest, `SourcePlacement`, `AudioWarpMap`, `EditMap` и двухфазного planner до переноса UI и Forge.
4. Фаза 1: characterization-тесты → core contracts → worker/runner → scan/prepare → complex sync → FCPXML. Только после прохождения этой вертикали добавлять режимы, Premiere и LLM.

---

## 11. Приложения: форматы

### 11.1. `project.json` (сокращённо)

```json
{
  "version": 2,
  "schema_version": 2,
  "producer_version": "studio-dev",
  "source_dir": ".",
  "assets": [
    {"id": "dji-0838", "path": "camA/DJI_0838.MP4", "kind": "video", "role": "camera",
     "device": "DJI Osmo Pocket 3", "group_id": "cam-a", "chapter_of": null,
     "created_at": "2026-09-30T18:02:11", "manual": {}},
    {"id": "zoom0001-tr1", "path": "ZOOM0001/ZOOM0001_Tr1.WAV", "kind": "audio",
     "role": "recorder", "device": "ZOOM H6", "group_id": "rec-1", "manual": {}}
  ],
  "cameras":   [{"id": "cam-a", "name": "Камера A", "asset_ids": ["dji-0838"], "speaker": "S1", "is_wide": false}],
  "recorders": [{"id": "rec-1", "name": "ZOOM H6", "asset_ids": ["zoom0001-tr1", "zoom0001-tr2"],
                 "tracks": [{"index": 1, "channels": [0], "speaker": "S1"}, {"index": 2, "channels": [0], "speaker": "S2"}]}],
  "placements": [
    {"asset_id": "dji-0838", "offset_s": 0.0, "in_s": 0.0, "duration_s": 1532.4, "k": 1.0, "provenance": "metadata"},
    {"asset_id": "zoom0001-tr1", "offset_s": -12.731, "in_s": 0.0, "duration_s": 1601.0, "k": 1.000062,
     "provenance": "acoustic", "evidence": {"inliers": 41, "residual_ms": 6.2, "coverage": 0.93}}
  ],
  "audio_warp_maps": [
    {"id": "warp-rec-1-cam-a", "source_asset_id": "zoom0001-tr1", "target_asset_id": "dji-0838",
     "strategy": 3, "time_domain": "rendered_audio", "pieces": [
       {"source_start_s": 0.0, "source_duration_s": 30.0, "rendered_start_s": 0.0,
        "rendered_duration_s": 29.998, "factor": 1.00007, "method": "resample"}
     ]}
  ],
  "transcripts": {"zoom0001-tr1": "stages/transcribe/zoom0001-tr1.json", "timeline": "stages/timeline/transcript.json"},
  "speaker_names": {"S1": "Иван", "S2": "Мария"},
  "edit": "stages/roughcut/edit.json",
  "stage_states": {"sync": {"fingerprint": "3f9c…", "status": "ok", "finished_at": "2026-10-05T21:14:03"}}
}
```

### 11.2. События исполнителя (JSON-lines в stdout процесса этапа)

```json
{"t": "start",    "stage": "sync"}
{"t": "progress", "stage": "sync", "value": 0.42, "message": "Клип DJI_0839: рендер голоса"}
{"t": "timeline", "stage": "sync", "clips": [{"asset_id": "dji-0839", "offset_s": 1532.4, "status": "working", "speed": 0.0011}]}
{"t": "warning",  "stage": "sync", "text": "GX010024: мало якорей, взят акустический сдвиг"}
{"t": "artifact", "stage": "sync", "path": "stages/sync/voice/dji-0839_voice.wav"}
{"t": "done",     "stage": "sync", "status": "ok", "output": "stages/sync/output.json"}
```

Ход процесса пишется в лог отдельно (stderr → `logs/`). В stdout — только события, чтобы их всегда можно было разобрать.

### 11.3. `edit.json`

```json
{
  "mode": "cut",
  "keep": [{"start": 3.20, "end": 61.84, "camera_id": "cam-a"}, {"start": 62.91, "end": 140.02, "camera_id": "cam-b"}],
  "cuts": [
    {"start": 0.0,   "end": 3.20,  "reason": "head",   "confidence": 1.0,  "note": "до первого слова", "accepted": true},
    {"start": 61.84, "end": 62.91, "reason": "pause",  "confidence": 0.95, "note": "пауза 1.47 с → 0.40 с", "accepted": true},
    {"start": 88.10, "end": 94.55, "reason": "retake", "confidence": 0.8,  "note": "дубль 1 из 2", "accepted": true}
  ],
  "markers": [{"at": 120.3, "kind": "filler", "text": "«как бы»"}]
}
```

### 11.4. Мультикам в FCPXML (скелет)

```xml
<resources>
  <format id="r1" frameDuration="1/25s" width="1920" height="1080"/>
  <asset id="r2" src="camA/DJI_0838.MP4" .../>
  <asset id="r3" src="camB/GX010024.MP4" .../>
  <asset id="r4" src="_studio/stages/sync/voice/dji-0838_voice.wav" .../>
  <media id="r10" name="Эпизод — мультикам">
    <multicam format="r1">
      <mc-angle name="Камера A" angleID="camA"><asset-clip ref="r2" offset="0s" duration="…"/></mc-angle>
      <mc-angle name="Камера B" angleID="camB"><gap offset="0s" duration="…"/><asset-clip ref="r3" lane="1" offset="…" duration="…"/></mc-angle>
      <mc-angle name="Голос"    angleID="voice"><asset-clip ref="r4" offset="0s" duration="…"/></mc-angle>
    </multicam>
  </media>
</resources>
<!-- в spine проекта: по mc-clip на каждый KeepRange -->
<mc-clip ref="r10" offset="0s" start="80/25s" duration="1466/25s">
  <mc-source angleID="camA"  srcEnable="video"/>
  <mc-source angleID="voice" srcEnable="audio"/>
</mc-clip>
```

### 11.5. XML для Premiere (xmeml v4, скелет)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE xmeml>
<xmeml version="4">
  <sequence id="seq-1">
    <name>Эпизод — черновик</name>
    <rate><timebase>25</timebase><ntsc>FALSE</ntsc></rate>
    <media>
      <video>
        <format><samplecharacteristics><width>1920</width><height>1080</height></samplecharacteristics></format>
        <track>
          <clipitem id="ci-1">
            <name>DJI_0838</name>
            <enabled>TRUE</enabled>
            <start>0</start><end>1466</end><in>80</in><out>1546</out>
            <file id="f-1"><pathurl>file://localhost/…/camA/DJI_0838.MP4</pathurl></file>
            <link><linkclipref>ci-1</linkclipref></link>
            <link><linkclipref>ca-1</linkclipref></link>
          </clipitem>
        </track>
      </video>
      <audio>
        <track>
          <clipitem id="ca-1">
            <file id="f-2"><pathurl>file://localhost/…/dji-0838_voice.wav</pathurl></file>
            <start>0</start><end>1466</end><in>80</in><out>1546</out>
            <sourcetrack><mediatype>audio</mediatype><trackindex>1</trackindex></sourcetrack>
          </clipitem>
        </track>
      </audio>
    </media>
    <marker><name>Пауза 1.47 с → 0.40 с</name><in>1466</in><out>-1</out></marker>
  </sequence>
</xmeml>
```

Скелеты 11.4 и 11.5 — ориентир для реализации. Точный набор обязательных атрибутов сверяется с реальным импортом в фазе 2 (чек-лист 8.4).
