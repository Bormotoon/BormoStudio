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
| 3 | Паузы/дубли, решения, обзор резов на таймлайне, прослушивание без мастера, предпросмотр всего монтажа (прокси), CFR-мастер с кроссфейдами есть | Реальное прослушивание, проверка слов на стыках, сквозной переэкспорт |
| 4 | Mics/pyannote facade, cross-file identity, выбор камер (GUI/CLI), политика общего плана, дорожки спикеров и multicam есть | Модели, проверка сопоставления на записи и NLE приёмка |
| 5 | Тексты/term-check, role routing, рилсы/аудиокэш, captions/editor, vision, fetch, channel/batch с уведомлениями и отменой загрузок, миграция настроек есть | Clip tiers/judge context/diversity и прочие ключи, phase-wide reuse и Forge-регрессия |
| 6 | Модули, загрузка YuNet/Light-ASD/Whisper/GGUF, LLM-детекторы монтажа, мастер настройки и Linux frozen есть | Приёмка LLM-монтажа на модели, GPU-прогон автонастройки, production installers |
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
- [ ] **Частично:** Qt-экраны/bridge, запуск/лог/отмена/Material и Settings dirty guards есть; полный предпросмотр монтажа — кэшируемый 360p-прокси (`uvf preview`) с плеером, курсором на обзоре резов и переходом по клику. Не доделано: drag/drop и визуальная/клавиатурная приёмка.
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

- [ ] **Частично:** mics/smoothing/pyannote facade есть; атрибуция микрофонов при сложной синхронизации идёт через кусочную `AudioWarpMap` (вне её покрытия — линейное размещение). Cross-file identity: метки разных файлов/сессий сводятся по голосу (эмбеддинги pyannote или MFCC-подпись, взаимно лучшая пара; одновременные файлы и разные ручные имена не сливаются), pyannote размечает каждый источник новой сессии. Не доделано: turns/manual naming parity и real model acceptance.
- [x] speaker-camera mapping задаётся на экране «Материал» и `uvf cameras` (хранится в проекте, перекрывает `program.speaker_cameras`); общий план назначается так же (`@wide`, «Wide shot»).
- [ ] **Приёмка:** правила 5.6 реализованы: camera_id, minimum shot, coverage fallback, manual precedence, наложения и частые перебросы → общий план, вставка общего в длинный план (`max_shot_s`), упреждение реза `cut_lead_s`. Не проверена реальная оценка переключений.
- [ ] **Приёмка:** retimed multicam/voice/ambience/speaker selection и XMEML lanes реализованы, generated FCPXML прошёл 1.9 DTD. Не проверены импорт/переключение/playback в NLE.
- [x] `program` учитывает выбранную камеру, ручные решения и speaker-camera mapping с проверкой покрытия.

**Готово, когда:** запись «2 камеры + 2 петлички» даёт в Final Cut мультикам с разумными переключениями, и ракурс на любом куске меняется штатно (клавишами или в инспекторе).

### Фаза 5. Тексты, рилсы и совместимость Forge

**Задачи:**

- [ ] **Частично:** provider/cache/session/retries/role routing есть; `tools/migrate_settings.py` переносит старые настройки по таблице 6.3 с отчётом по каждому ключу. Разделы `fetch` (формат, качество, cookies, повторы, скорость, фильтры канала) и `batch` (уведомления, минимум свободного места) работают и принимают ключи `youtube.*`/`autonomy.*`; заменённые ключи получают в отчёте конкретную замену. Ключи llama service tuning, watchdog, параметры по ролям и subtitle styling переносятся. Не доделано: clip tiers, judge context/diversity, subtitle whisper_sync и phase-wide model reuse.
- [x] Вычитка/realignment, статья и term-check интеграция реализованы: ручные offline fixes, отдельный network opt-in, кэш и отчёт. Реальная приёмка отдельно.
- [ ] **Частично:** verified selection/refine/judge/ranking/context, role routing и content-based audio probes/cache реализованы. Voice/placement/edited mapping позволяют работать без постоянного мастера/program. Без voice WAV анализ воспроизводит голос из сырого рекордера через `AudioWarpMap` (raw warp replay). Не доделано: полный legacy settings и moments golden parity.
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
- [ ] **Приёмка:** modules manager/recipes/GUI/progress/cancel/locks есть; планировщик проверяет требования каждого запускаемого этапа (ffmpeg, faster-whisper, модули, модели, управляемый llama), `uvf plan`/`run` называют недостающее и как установить, `run` не запускает этап без обязательного, экран плана ведёт кнопкой в «Модули и модели» с предвыбранной позицией. Не проверены реальные ML установки/запуски.
- [x] YuNet/Light-ASD, Whisper (base…large-v3-turbo) и Gemma 4 GGUF загружаются из приложения: закреплённые ревизии, проверка хэшей, докачка, отмена, подстановка в настройки; мастер первого запуска показывает скачанное, предлагает large-v3-turbo и сохраняет выбранные Whisper/GGUF в проект. Реальная загрузка на чистой машине — приёмка фазы.
- [ ] **Частично:** Linux frozen, setup wizard и local install script (binary/link/desktop entry) есть; same-host installed smoke пройден. Не доделано: dependency bundling, clean-machine/Python 3.12 и Windows/macOS; бинарник требует rebuild после последних guards.
- [ ] **Приёмка:** compute probe и измеряемая автонастройка (`uvf tune`, кнопка в настройках): Whisper тип × batch и llama слои × batch/ubatch в отдельных процессах, пик видеопамяти через nvidia-smi, рекомендации с запасом памяти. Реальный CPU-прогон пройден; GPU-прогон не выполнялся.

**Готово, когда:** на чистой машине установщик → первый запуск → проект из фикстур проходит без ручной установки чего-либо, кроме драйвера NVIDIA.

### Фаза 7. Дальше (по приоритету)

- [ ] **Частично:** source-video reels есть; рилсы из таймлайна переключают ракурс по спикерам той же политикой камер, что программа (`reels.speaker_angles`). Per-reel framing (`uvf reel`, «Только этот рилс» в редакторе) и мгновенный предпросмотр кадра по неотрезанному кадру рилса сделаны; визуальная приёмка отдельно.
- [ ] **Приёмка:** vision авто-привязка камер к спикерам реализована (окна реплик → YuNet + Light-ASD в vision-модуле → контраст «говорит в своих репликах», общий план по ≥2 лицам; `uvf cameras --suggest [--apply]`, кнопка на «Материале», ручной выбор не перезаписывается). Не запускалась на реальных моделях и записи.
- [ ] **Приёмка:** OTIO-экспорт (`export.targets: otio`) реализован без новых runtime-зависимостей и читается OpenTimelineIO 0.18; импорт в DaVinci Resolve не проверен.
- [ ] **Приёмка:** MLX Whisper (`transcribe.device=mlx`, extra `[mlx]`) и VideoToolbox (`program.encoder=videotoolbox`, auto на macOS с откатом на libx264) реализованы и покрыты тестами на подменах; на Mac не запускались.
- [ ] **Частично:** нативный subtitle editor есть; последующие расширения/визуальная приёмка остаются.
- [ ] **Частично:** −14 LUFS (двухпроходный loudnorm), обложки (кадр рилса) и YouTube most-replayed (heatmap из info.json как фактор ранжирования) реализованы. Hook-title layout и аудиограмма (`reels.layout`) и Forge-пресеты субтитров (`reels.caption_preset`) реализованы; визуальная приёмка отдельно.

### Исправления по реальному прогону (pos17v2, 2026-10-10)

Материал: `/srv/storage/files/pos17v2/` — камера `DJI_0909.MOV` (4 мин 52 с, 1080p 29,97) и первые 30 мин рекордера `2026-10-05_103823465.wav` (стерео 48 кГц, полная запись 4 ч 29 мин). Все этапы до экспорта прошли без ошибок (≈3 мин на RTX 5060 Ti); камера размещена на 47,8 с рекордера, участок с речью сведён с точностью 0–1 мс. Ниже — найденные дефекты по приоритету.

- [ ] **Проверка синхронизации пропускает частичный рассинхрон.** `uvf verify`, самопроверка и `sync.self_check=repair` выносят вердикт по медиане |lag|: на этом клипе 0,8 мс → `passed`, хотя p90 = 85 мс и первые ~62 с клипа сдвинуты на −88 мс (≈2,6 кадра). Сделать: проваливать проверку, если доля уверенных окон с |lag| выше `sync.max_drift_ms` превышает порог (или по p90), и называть в отчёте сдвинутые интервалы. Готово, когда этот клип получает `failed` с интервалом ≈0–62 с, а клипы с единичным ложным окном по-прежнему проходят. → Решение Р1. Реализовано (`fda8b9d`, ремонт через Р2 — `7c9bcf5`). Приёмка на реальном материале — отдельно.
- [ ] **Скачок синхронизации внутри клипа не обнаруживается.** Сдвиг постоянный −88 мс до 62,5 с клипа и ≈0 после; на этом месте (110,1 с рекордера) в рекордере резкий щелчок — вероятно, рекордер выронил ~88 мс записи (метки времени камеры непрерывны, нулевых участков нет; доказать нельзя: похожие щелчки на 117,7 и 177,7 с скачка не дают). Режим `auto` выбрал одну постоянную поправку (стратегия 1, 3 опорные точки). Сделать: акустическое уточнение по сетке окон (GCC-PHAT, как в `verify`) после выбора стратегии; при устойчивой ступеньке — разрыв в `AudioWarpMap` (тишина/обрезка в месте щелчка) с записью в отчёт. Готово, когда на этом клипе расхождение ≤ 5 мс по всей длине, а в отчёте указан разрыв на ≈110,1 с рекордера. → Решение Р2. Реализовано (`7c9bcf5`, `0317bdc`). Приёмка на реальном материале — отдельно.
- [ ] **`sync.mode=complex` даёт ложный дрейф.** Текстовая подгонка дала k = 1.0000708 (≈4 мс/мин) и 29 кусков, но скачок не нашла, а расхождение на речи выросло до 10–22 мс (выше `max_drift_ms` 20 мс); постоянная поправка `auto` на том же участке даёт 0–1 мс. Вероятно, наклон смещён разрывом и неточными словами. Сделать: проверять подгонку акустикой и отклонять дрейф, не подтверждённый окнами; не принимать k, ухудшающий p90 относительно постоянной поправки. Готово, когда `complex` на этом клипе не хуже `auto`. → Решение Р3. Реализовано (`3222954`). Приёмка на реальном материале — отдельно.
- [ ] **Транскрибируется весь рекордер, даже если камеры покрывают малую часть.** Для 5-минутного клипа камеры Studio начала транскрибировать все 4,5 ч рекордера. Сделать: транскрибировать рекордер только вокруг возможного покрытия камер (по метаданным времени с запасом, затем расширять при неудаче) и/или настройку диапазона (`transcribe.max_minutes` / интервал по ассету). Часы камеры здесь отстают от имени файла рекордера на ≈45 мин стабильно в начале и в конце дня — это можно использовать как подсказку окна поиска после первого найденного сдвига. → Решение Р4. Реализовано (`5de2f9f`). Приёмка на реальном материале — отдельно.
- [ ] **Черновой монтаж оставляет подготовку до первой фразы.** Вырезано 41 с из 30 мин; кусок 0–187 с (подготовка, речь начинается на 186,3 с) оставлен целиком, потому что паузы с нераспознанным звуком громче −40 дБ не режутся (маркеры `unknown-sound`). Сделать: отдельное правило для начала/конца записи — до первого и после последнего распознанного слова нераспознанный звук режется предложением (с маркером, ручным возвратом и прослушиванием), а внутри речи остаётся как сейчас. Готово, когда на этом материале голова 0–186 с предлагается к удалению, а внутренние «unknown-sound» не трогаются. → Решение Р5. Реализовано (`0fda3f0`). Приёмка на реальном материале — отдельно.
- [ ] **Неверная метка слова через долгую тишину (быстрый режим Whisper).** В транскрипте звука камеры слово «По» стоит на 14,8 с, следующее — на 141,7 с (сегмент 154 с). Основной транскрипт таймлайна (рекордер) корректен, но транскрипт камеры используется для сопоставления и самопроверки. Сделать: при пропуске между соседними словами больше N секунд внутри сегмента пере-выравнивать слово (VAD/акустика) или выкидывать его из опорных точек; добавить проверку в `transcribe`. Готово, когда в транскриптах нет слов, отделённых от своей фразы минутами тишины. → Решение Р6. Реализовано (`917cc02`). Приёмка на реальном материале — отдельно.
- [ ] **Сообщение при `--only` после смены режима синхронизации непонятно.** `uvf run --set sync.mode=complex --only sync …` блокирует все этапы с «discovery inputs are not finalized»: смена режима меняет отпечаток `prepare`, а он исключён. Сделать: называть причину («prepare устарел из-за sync.mode; добавьте prepare или уберите --only») или автоматически включать устаревший `prepare` в `--only`. → Решение Р7. Реализовано (`6c866be`). Приёмка на реальном материале — отдельно.
- [ ] **Двойной моно рекордера не распознаётся.** Оба канала рекордера идентичны (одна петличка, продублированная в стерео); это не две дорожки спикеров. Сделать: при сканировании отмечать совпадающие каналы (корреляция ≈1) в `media_info`/`RecorderTrack`, не предлагать их как два спикера и декодировать как моно. Готово, когда этот файл показывается одной дорожкой. → Решение Р8. Реализовано (`b24aac6`). Приёмка на реальном материале — отдельно.
- [ ] **Долгий старт на больших исходниках.** Перед транскрипцией процесс заметное время читает 3-гигабайтный WAV (sha256 для манифестов/отпечатков). Сделать: кэшировать дайджест источника по (путь, размер, mtime) в рабочей папке проекта, пересчитывать только при изменении. → Решение Р9. Реализовано (`4e2ecc6`). Приёмка на реальном материале — отдельно.

#### Решения (проработано 2026-10-10)

> **Статус на 2026-10-10:** решения «А», Р1–Р9 реализованы, покрыты тестами (616 проходят) и прогнаны сквозь конвейер на синтетических сценариях `tools/make_fixtures.py --defects` (Whisper small, RTX 5060 Ti): выпадение 88 мс найдено по щелчку и поглощено в паузе (verify: p90 < 0,01 мс), двойной моно — одна дорожка и моно-голос, шумная голова 0–60,5 с предложена к резу с уверенностью 0,7, 2-минутный клип в 20-минутном рекордере — распознано 10 из 20 мин (2 плитки), слово после шумового всплеска стоит у своей фразы. Повторная приёмка на `pos17v2` не выполнялась: материала нет в `/srv/storage/files/pos17v2/`.
>
> Отступления от решений:
> - Р2: маркер `sync-step` в экспорте (опциональный) не сделан.
> - Р3, п.2: принудительных точек разреза на границах участков внутри стратегии 3 нет — сглаживание не смешивает якоря разных участков, а ступеньку затем поглощает уточнение Р2, которое работает после любой стратегии.
> - Р6, п.2: старый кэш транскриптов не перепроверяется, а пересоздаётся (смена `CACHE_SCHEMA` и версии сегментации в отпечатке).
> - Р8: у дорожки рекордера появился флаг `manual` — без него ручное разделение каналов нельзя отличить от старого предложения «по дорожке на канал».
> - Р9: правило гонки применяется только к файлам с метками времени, кратными секунде (грубые ФС); на ext4/NTFS/APFS ему не с чем работать.
> - Таблица миграции не менялась: у новых настроек нет аналогов в WhisperSync и Forge.

Анализ выполнен по коду `studio/` на `d75a338` и по артефактам прогона (`_studio/stages/sync/{output,self-check,repair}`, транскрипты, `roughcut/edit.json`). Сквозные выводы:

- Звук камеры с голосом сейчас сравнивают три независимых механизма с разными правилами: `sync_verify.measure` (окна каждые 5 с, вердикт по медиане), Boundary Flex (`refine_piece_boundaries`: только начала кусков, по умолчанию выключен) и `acoustic_coarse_align` (окна каждые 30 с, только для размещения). Р1–Р3 строятся на одном общем компоненте «А».
- Длинные исходники за прогон читаются целиком много раз: SHA-256 в отпечатках при каждом перепланировании, полный декод рекордера в `pcm_f64le` (`selected_media`) и в 16 кГц для каждой пары камера–рекордер, Whisper по всему файлу. Это закрывают Р4 и Р9.
- Отпечаток `prepare` — хэш всех настроек, поэтому любой `--set` делает обнаружение устаревшим. Тем же дефектом сломана кнопка рендера субтитров в GUI (Р7).

#### А. Общий компонент: акустическая сетка и участки

`studio/stages/sync_grid.py`, используется в Р1–Р4.

- `measure_grid(camera, other, mapping, grid_s=2.5, window_s=4, max_lag_s=0.25, min_sharpness=50) -> VerifyReport`: окно звука камеры в момент `t` сравнивается `gcc_phat` с окном второй дорожки в момент `mapping(t)`. Для отрендеренного голоса `mapping` — тождество (нынешний `measure`), для плана до рендера — карта кусков «время клипа → время рекордера». Каждая дорожка декодируется в моно 16 кГц один раз и только в нужном диапазоне (`load_mono16k_track(start_s, duration_s)`).
- Знак один на весь код: `residual(t)` — на сколько позже (+) или раньше (−) нужно читать рекордер в момент `t`. Синтетический тест с известным сдвигом закрепляет знак для сетки, коррекции и отчёта.
- `segment_regimes(samples, step_min_ms=10, min_windows=3, max_mad_ms=5)`: робастное деление остатков на участки — бинарное деление по скачку медиан соседних групп. У участка ≥ 3 уверенных окон и разброс (MAD) ≤ 5 мс; группы короче трёх окон считаются выбросами. Линейный тренд внутри участка допускается, только если улучшает p90 участка ≥ 2 мс.
- Выбор модели: «константа», «прямая» или «участки» (с трендом или без). Более сложная модель берётся, только если её предсказанный p90 лучше ≥ 2 мс. Результат — `RegimeModel`: участки `(start_s, end_s, offset_ms, slope)`, ступеньки и p90 до/после коррекции.

#### Р1. Вердикт проверки синхронизации

**Причина.** `VerifyReport.verdict` (`sync_verify.py`) проваливает клип только по медиане |lag|, а сдвинутая треть клипа медиану не меняет. Порог 20 мс зашит в `sync.py:353` и `sync_repair.py:58` вместо `sync.max_drift_ms`. Самопроверка (`sync_check_step.check_voices`) сравнивает только слова Whisper с порогом 0,35 с (`DEFAULT_SHIFT_THRESHOLD_S`) и в принципе не видит 88 мс. Ремонт (`repair_voices`) перерисовывает клип одной прямой (strategy 1), а ступеньку так не исправить. На этом клипе его кандидат к тому же отклонён по ложному content-провалу: на 251–262 с Whisper не распознал речь в звуке камеры, хотя голос там есть.

**Решение.**
1. `verdict(threshold_ms, min_bad_windows=3, max_bad_fraction=0.1)`. Плохое окно — уверенное окно с |lag| > порога. Отброшенные по резкости окна не прерывают серию плохих, если соседние уверенные окна ближе 3·`grid_s`. Серия из ≥ 3 плохих окон — сдвинутый интервал; его границы — середины между крайними плохими и соседними хорошими окнами (или края клипа). `failed`, если есть такой интервал или доля плохих окон > 10 %. Серии из 1–2 окон попадают в `outliers` и вердикт не меняют. Проверки покрытия (`inconclusive`) остаются прежними.
2. `summary()` дополняется полями `shifted_intervals: [{start_s, end_s, median_lag_ms, windows}]`, `outliers`, `bad_fraction` и `threshold_ms`; `reason` перечисляет интервалы («0,0–64,5 с: −88 мс, N окон»).
3. Порог везде — `sync.max_drift_ms`: в `sync.verify`, в ремонте и в самопроверке. `uvf verify --threshold-ms` по умолчанию берёт его из проекта; добавляются `--min-bad-windows` и `--max-bad-fraction`.
4. Самопроверка (`warn`/`repair`) дополняется акустикой: для каждого голоса — `measure` и новый вердикт. Клип `failed`, если провалена акустика или подтверждённый content. Content-интервал, где акустические окна уверенные и в пределах порога, понижается до предупреждения `camera_transcript_gap`: речь на месте, ошибся Whisper по звуку камеры.
5. Ремонт строит кандидата через Р2 (участки и ступеньки), а не одной прямой; в `align_sources` передаются настройки `sync` вместо `{}`. Кандидат принимается при акустическом `passed` и content не хуже исходного.
6. `sync.verify` по умолчанию `true`: измерение дешёвое по сравнению с Whisper, а при включённом Р2 сетка всё равно считается.

**Тесты:**
- голос со сдвигом 88 мс на первых 60 с из 290 → `failed` с интервалом ≈0–62 с;
- одно ложное окно 120 мс → `passed` и один `outlier`;
- рассеянные 15 % плохих окон → `failed` по доле;
- content-провал при хорошей акустике → `camera_transcript_gap`.

**Готово** — как в пункте. Проверяется на старом голосе `dji-0909.wav` или при `sync.acoustic_refine=off`; с Р2 тот же клип должен проходить.

#### Р2. Ступенька внутри клипа

**Причина.** Все стратегии строят непрерывную карту без ступенек. Стратегия 1 — одна прямая. Стратегия 3 режет по предложениям, но голова до первой фразы (здесь 0–141 с камеры) — один кусок (`sync_pieces._sentence_pieces`). Boundary Flex двигает только начала кусков, не больше чем на 0,1 с, и по умолчанию выключен. `acoustic_coarse_align` берёт окна раз в 30 с и только для размещения.

**Решение** — акустическое уточнение плана после выбора стратегии: `studio/stages/sync_refine.py`, вызов в `render_aligned_clip` между `clip_pieces` и рендером, до Boundary Flex.
1. Сетка «А» по плану до рендера (камера ↔ рекордер через карту кусков) и `segment_regimes`. Если участок один и |offset| ≤ 3 мс, план не меняется.
2. Граница каждой ступеньки уточняется между последним окном участка A и первым окном участка B:
   - мелкая сетка (окно 1 с, шаг 0,1 с): место, где лаг переходит от A к B;
   - щелчок в рекордере на этом отрезке: энергия выше 2 кГц кадрами 10 мс, скачок ≥ 20 дБ над локальной медианой; берётся, если он в пределах ±1 с от оценки мелкой сетки;
   - если не вышло ни то ни другое — середина между окнами с пометкой `coarse`.
3. Сбой не в камере: проверяется непрерывность PTS аудиопакетов камеры вокруг ступеньки (`ffprobe -read_intervals … -show_packets`). При разрыве в камере ступенька не исправляется, в отчёт пишется предупреждение.
4. Коррекция использует ту же схему перенацеливания границ, что `refine_piece_boundaries._build`, но δ берётся из модели, а не из локальных замеров:
   - выходные окна кусков не меняются;
   - куски делятся по окну поглощения `[b − w/2, b + w/2]`, источник каждого куска сдвигается на δ(t) своего участка;
   - кусок поглощения сохраняет выходную длину `w`, его источник становится `w + Δ` (темп `(w + Δ)/w`);
   - `w = max(0,3 с, |Δ|/0,2)`, окно переносится в ближайшую паузу между словами рекордера (`recorder_word_gaps`, в пределах `seam_snap_max_s`);
   - если паузы рядом нет, `w = |Δ|/0,03` (≈3 с для 88 мс) и предупреждение «поглощено в речи».
   Контракт `AudioWarpMap` не меняется: обе шкалы непрерывны, план проходит `validate_pieces`, темп в [0,5; 2]. Дыру в карте, предложенную в пункте, делать не нужно: §4.3 требует покрытия без дыр, и на это опираются `_validate_warp_pieces`, экспорт, `speakers` и raw warp replay. Короткий кусок поглощения в паузе на слух неотличим от вставки тишины.
5. Уточнение принимается, если предсказанный p90 после коррекции ≤ порога и лучше исходного хотя бы на 2 мс. Иначе план прежний и в отчёте `refine rejected`. После рендера окончательное решение — вердикт Р1.
6. Отчёт `stages/sync/output.json → acoustic_refine[camera]`: участки, ступеньки (`camera_s`, `recorder_s`, `delta_ms`, `located_by`: click|fine|coarse, окно поглощения), p90 до и после, `accepted`. Событие `warning`: «DJI_0909: скачок −88 мс на 110,1 с рекордера, исправлен». `AudioWarpMap.evidence` получает `steps` и `refine_p90_ms`. Опционально — маркер `sync-step` в экспорте.

**Настройки:** `sync.acoustic_refine: auto|off` (auto — когда у камеры есть звук), `sync.step_min_ms: 10`, `sync.refine_grid_s: 2.5`, `sync.refine_window_s: 4`.

**Тесты:**
- рекордер с выпавшими 88 мс и щелчком → ступенька в пределах ±0,25 с от щелчка, расхождение ≤ 5 мс по всей длине, карта непрерывна;
- ступенька посреди речи → широкое поглощение и предупреждение;
- разрыв PTS в камере → без исправления;
- клип без ступеньки → план не меняется.

#### Р3. Ложный дрейф в `complex`

**Причина.** В `complex` размещение и `k` берутся из текстовых якорей (`sync_matcher.align`, RANSAC по словам Whisper с точностью ±50–100 мс). Ошибка k = 7·10⁻⁵ на 292 с (≈20 мс за клип) по ним неотличима от шума. Стратегия 3 сглаживает те же якоря (`_smoothed_map_at`, ±30 с), и каждое предложение получает свою ошибку 10–22 мс. Акустикой подгонка не проверяется. У акустических карт `residual_ms` всегда 0 (`sync_alignment.py:37`), поэтому `choose_sync_mode` не может честно сравнить кандидатов.

**Решение.**
1. После `plan_sessions` для каждой пары камера–рекордер строится сетка «А» вдоль принятой прямой. Если уверенных окон ≥ 8 и они покрывают ≥ 50 % клипа, `offset` и `k` берутся из акустической модели («константа», «прямая» или «участки» по правилу «А»). Текст остаётся грубым размещением и запасным путём для камер без звука; текстовый `k` используется только без акустики.
2. Якоря стратегий 2/3 — акустические: каждое уверенное окно даёт `Anchor(rec_time, cam_time)`. Текстовые якоря добавляются только там, где на ±10 с нет акустических. `_smoothed_map_at` не смешивает якоря разных участков. Границы участков становятся принудительными точками разреза с поглощением как в Р2, голова тоже делится.
3. `residual_ms` акустической карты — p90 остатка сетки, а не 0.
4. Защита «не хуже `auto`»: в `complex` по той же сетке считается предсказанный p90 плана `auto` (константа + участки). Если `complex` хуже больше чем на 1 мс, клип рендерится планом `auto`, а в отчёте пишется: «complex отклонён: дрейф не подтверждён акустикой (k_text = 1,0000708; p90 22 мс против 1 мс)».

**Тесты:**
- без дрейфа, текстовые якоря с шумом ±80 мс → `k = 1`, p90 ≤ 2 мс;
- реальный дрейф 1,00075 (`make_fixtures`) → дрейф подтверждён акустикой;
- ступенька в режиме `complex` → результат как в Р2.

**Готово** — как в пункте; выбранная модель и судьба текстового `k` видны в `output.json`.

#### Р4. Транскрипция рекордера только там, где есть камеры

**Причина.** `prepare` всегда требует транскрипт всего рекордера (и всех камер), а `transcribe` гонит Whisper по всему файлу. Время записи (`creation_time` камеры, имя файла рекордера, BWF, mtime) нигде не используется. Sync тоже обрабатывает рекордер целиком:
- `selected_media` декодирует каждый исходник в `pcm_f64le` (4,5 ч стерео 48 кГц ≈ 12 ГБ на прогон);
- `acoustic_coarse_align` и Boundary Flex заново декодируют весь рекордер в 16 кГц для каждой пары;
- `acoustic_coarse_align` держит в памяти спектры всех блоков (≈2,4 ГБ на 4,5 ч).

**Решение.**
1. `scan` пишет `media_info.recorded_at = {start, source, precision}`. Источники: `creation_time` (QuickTime/DJI — начало записи, UTC), BWF `bext`, имя файла (`YYYY-MM-DD_HHMMSS[mmm]`, `YYYYMMDD_HHMMSS`), иначе `mtime − duration`.
2. Новый этап обнаружения `locate`: после `scan`, до `prepare`, только CPU; добавляется в `DISCOVERY_STAGES` и в `uvf scan/plan`. Для каждой группы камер и рекордера:
   - выбирается опорная камера — самая длинная со звуком;
   - 3 её окна по 8 с ищутся по всему рекордеру блочной корреляцией из `acoustic_coarse_align`, но потоково: рекордер декодируется кусками через ffmpeg pipe, и каждый блок после корреляции освобождается, так что память O(блок);
   - по найденной позиции считается смещение часов группы `delta` (на pos17v2 ≈45 мин);
   - остальные камеры группы ищутся в ±`sync.locate_search_s` (120 с) от `recorded_at + delta`; при неудаче — ±15 мин, затем по всему рекордеру; камеры без `recorded_at` сразу ищутся по всему рекордеру;
   - результат кэшируется по ключам дайджестов Р9; найденная `delta` подсказывает поиск следующим запускам.
   Выход — `stages/locate/positions.json`: `{camera: {recorder, position_s, method, sharpness, delta_s}}`.
3. `prepare` пишет в `requirements.json` поле `transcribe_ranges[recorder]`:
   - объединение `[позиция − margin, позиция + длительность + margin]` по камерам, `margin = transcribe.window_margin_s` (60 с); диапазоны с промежутком < 2·margin сливаются;
   - весь файл берётся, если объединение ≥ 80 % файла, камер нет или какую-то камеру не нашли (причина — в `reasons`). Так «расширение при неудаче» происходит до построения DAG, без скрытого цикла;
   - транскрипт камеры нужен, только если её не нашёл `locate`, включён `self_check`, режим `camera` или рекордера нет (`transcribe.camera_transcripts: auto|always`); длительность клипа в sync берётся из probe, а не из транскрипта.
4. `transcribe` режет диапазоны на плитки по сетке файла: 300 с с перекрытием 10 с, слово достаётся плитке, в ядро которой попала его середина. Каждая плитка декодируется через `-ss/-t` и распознаётся отдельно. Ключ кэша — (дайджест, поток, плитка, настройки), поэтому новая камера добавляет только новые плитки. Транскрипт хранит `duration` всего файла (из probe) и `metadata.coverage` — покрытые диапазоны во времени файла.
5. Потребители учитывают `coverage`:
   - `timeline` переводит его во время таймлайна;
   - `roughcut` ищет тишину, `unknown_sound_spans`, голову и хвост только внутри покрытия (`detect_silence` с `-ss/-t`);
   - если `sync` не нашёл якорей внутри покрытия, в `rejected_pairs` пишется «камера вне транскрибированного окна; transcribe.recorder_scope=full».
6. Декодирование в sync ограничивается нужным:
   - `acoustic_coarse_align`, Р2 и Boundary Flex декодируют только `[rec0 − 2 с, rec1 + 2 с]` клипа; декод одного рекордера переиспользуется для всех его камер (параметр `rec_track` уже есть);
   - `selected_media` не перекодирует PCM-WAV с одним аудиопотоком: рендер читает исходник напрямую;
   - остальное декодируется в `pcm_f32le` (24 бит без потерь) вместо `pcm_f64le` и только в нужных каналах (Р8).

**Настройки:** `transcribe.recorder_scope: auto|full|cameras`, `transcribe.window_margin_s: 60`, `transcribe.ranges: {asset_id: [[s, e], …]}` (ручной приоритет), `transcribe.camera_transcripts: auto|always`, `sync.locate_search_s: 120`.

**Тесты:**
- синтетический рекордер 30 мин и клип 2 мин в середине → ≤ 3 плиток, сдвиг найден;
- нет метаданных времени → полный поиск;
- камера не найдена → рекордер целиком, причина видна в плане;
- вторая камера не перераспознаёт готовые плитки;
- `coverage` доходит от `timeline` до `roughcut`;
- память `locate` на 4-часовом синтетическом рекордере ограничена.

**Готово:** на клипе DJI_0909 с полным рекордером 4,5 ч Whisper обрабатывает ≤ 10 мин рекордера, а камера стоит на тех же 47,8 с. На полном наборе (12 камер покрывают весь день) рекордер транскрибируется целиком, как и сейчас, а транскрипты камер — только для ненайденных.

#### Р5. Голова и хвост с нераспознанным звуком

**Причина.** `build_edit_list` режет голову и хвост, только если всё до первого (после последнего) слова — одна сплошная тишина ниже `silence_threshold_db` (`roughcut.py:302–307`). Один щелчок или шаг отменяет рез целиком.

**Решение.**
1. Голова — `[начало покрытия, первое надёжное слово − head_tail_pad_s]`. Покрытие — `audio_coverage` ∩ `coverage` транскрипта (Р4). Надёжное слово не помечено Р6 и имеет `probability ≥ 0,3`. Хвост — симметрично.
2. Тихая голова режется как сейчас (confidence 1,0). Голова со звуком — `Cut("head", 0.7)` с примечанием «до первого слова 186,3 с; нераспознанный звук: N фрагментов, M с». `accepted` задаёт `roughcut.head_tail_sound: cut|suggest|keep`. По умолчанию `cut`: рез применён и виден в обзоре как предложение с уверенностью 0,7. `suggest` — предложение без применения, `keep` — нынешнее поведение.
3. Маркеры `unknown-sound` внутри головы и хвоста остаются для прослушивания. Рез виден в обзоре резов с кнопкой «прослушать» и возвращается снятием галочки (`edit-overrides.json`) без пересчёта `roughcut`.
4. Конец реза прижимается к тишине перед первым словом: к началу последнего `silence`-интервала, который заканчивается не позже первого слова и начинается не раньше чем за 1 с до него. Защита «не внутри слова» сохраняется.
5. Внутренние паузы и правило `quiet()` для них не меняются.

**Тесты:**
- голова 180 с с шумами громче −40 дБ и первым словом на 186,3 с → рез 0–186,1 с с confidence 0,7, маркеры внутри сохранены;
- `suggest` → `accepted=false`;
- внутренняя пауза с unknown-sound не режется;
- тихая голова — confidence 1,0, как раньше.

#### Р6. Слово, оторванное от своей фразы

**Причина** (проверено по коду faster-whisper 1.2.1). В быстром режиме `BatchedInferencePipeline` склеивает фрагменты Silero VAD в куски до 30 с речи, не глядя на тишину между ними (`vad.collect_chunks`). Затем времена слов возвращаются по фрагменту, в котором лежит середина слова (`restore_speech_timestamps`). Режим `quality` с `vad_filter=True` склеивает так же. Шум на 14,77 с камеры VAD принял за речь, он попал в один кусок с фразой со 141 с, и слово «По» на стыке получило время шума. Тот же эффект есть в транскрипте рекордера: «Как» стоит на 455,5 с, а «думаете,» — на 509,9 с, в одном сегменте.

**Решение.**
1. Устранить причину в `FasterWhisperBackend._transcribe_once`: VAD вызывается отдельно (`faster_whisper.vad.get_speech_timestamps`, те же `VadOptions`), и в модель передаются свои клипы через `clip_timestamps`:
   - клип — непрерывный диапазон исходного аудио ≤ 30 с;
   - соседние речевые фрагменты сливаются, только если тишина между ними ≤ `transcribe.vad_merge_gap_s` (1,5 с);
   - тишина внутри клипа остаётся в аудио, поэтому времена слов истинные и восстанавливать их не нужно;
   - одиночные фрагменты короче 0,25 с отбрасываются;
   - в `quality` — тот же список клипов и `vad_filter=False`; MLX не затронут, VAD-склейки там нет.
2. Страховочная проверка `core/transcript_sanity.py` — для любого бэкенда, в `transcribe` и при чтении старого кэша:
   - сегменты делятся на паузах между словами длиннее `transcribe.max_word_gap_s` (2 с);
   - кандидат в «оторванные» — серия ≤ 2 слов, отделённая от продолжения фразы паузой длиннее этого порога. Признаки: после серии нет конца предложения и продолжение начинается со строчной буквы, либо серия — первое слово перед длинной паузой;
   - если вплотную перед продолжением есть речь без слов (по энергии), серия переносится туда с исходной длительностью;
   - иначе, если на исходном месте речи нет, слова получают `probability ≤ 0,1` и попадают в `metadata.suspect_words`;
   - иначе слова остаются на месте: это настоящая пауза.
   Слова с низкой вероятностью уже не становятся якорями (`normalize_words`, порог 0,6); Р5 не берёт их первым или последним словом.
3. `CACHE_SCHEMA` → `studio/transcript-cache/3`, в отпечаток `transcribe` входит версия сегментации; старые транскрипты пересоздаются.
4. Если подозрительные слова остались, `transcribe` выдаёт `warning`.

**Тесты:**
- построитель клипов: фрагменты 14,77–15,47 и 141,0–170 дают два клипа, ни один не включает тишину > 1,5 с;
- заглушка бэкенда со склейкой → «По» стоит у «новой,»;
- настоящая пауза 8 с между предложениями не трогается;
- «Как … думаете» переносится к продолжению фразы.

**Готово** — как в пункте; на этом материале «По» стоит на ≈141,2 с в транскрипте камеры, «Как» — на ≈509 с в транскрипте рекордера.

#### Р7. `--only` после смены настроек

**Причина.** Отпечаток `prepare` — хэш всех настроек (`prepare.py:28`), поэтому любой `--set` делает обнаружение незавершённым. `build_plan` выключает `prepare` через `--only` и блокирует остальное обезличенным «discovery inputs are not finalized». Так же сломан рендер субтитров из GUI: `MainWindow._render_subtitles` запускает `run --only <этап> --set …`.

**Решение.**
1. Каждый этап объявляет `settings_keys` — ключи, которые он читает; в отпечаток идут только они. Для `prepare` это `sync.mode`, `transcribe.audio_streams`, ассеты (id, роль, `manual`) и `plan_revision`; после Р4 — ещё `transcribe.recorder_scope/ranges/window_margin_s/camera_transcripts` и выход `locate`. Остальные этапы проходят тот же аудит: например, `timeline` сейчас хэширует весь раздел `sync`.
2. Этапы обнаружения (`scan`, `locate`, `prepare`) не подчиняются `--only`: устаревший этап планируется к запуску с причиной «выполняется автоматически: …». Явный `--skip` уважается, но тогда блокировка называет причину.
3. Объяснение устаревания. Манифест уже хранит `effective_settings`; к нему добавляются хэши частей отпечатка (`inputs.fingerprint_parts`: настройки, ассеты, транскрипты, артефакты). `planner.explain_stale(stage)` называет изменившуюся часть, а для настроек — разницу по ключам из `settings_keys`. Примеры сообщений:
   - «prepare устарел: sync.mode auto → complex»;
   - «sync пропущен: transcribe устарел (transcribe.model large-v3 → large-v3-turbo); добавьте transcribe в --only»;
   - с `--skip prepare`: «prepare устарел: sync.mode auto → complex; уберите --skip prepare».
   Причины пишутся в `report.json` (`stale_reasons`) и показываются на экране плана.

**Тесты:**
- `--set reels.layout=title --only reels` не трогает `prepare`;
- `--set sync.mode=complex --only sync` → автоматически `prepare`, затем `sync`; список транскриптов не изменился, поэтому `transcribe` переиспользуется;
- `prepare` поменял список транскриптов → `sync` пропущен с причиной «transcribe устарел: нужен транскрипт X»;
- командная строка GUI-рендера субтитров с `--set` планируется к выполнению;
- тесты `test_planner.py` с «discovery inputs are not finalized» переписываются под новые сообщения.

#### Р8. Двойной моно

**Причина.** `default_tracks` (`core/recorder_tracks.py:18`) предлагает по дорожке на канал, а содержимое каналов не сравнивается. `selected_media`, рендер голоса и экспорт сохраняют все каналы, поэтому голос выходит стереофайлом из двух одинаковых каналов.

**Решение.**
1. `scan` анализирует каналы аудиоассетов с ≥ 2 каналами и пишет результат в `media_info.channel_analysis`:
   - 4 отрывка по 5 с (на 10/35/60/85 % длительности) декодируются без сведения (8 кГц, f32 через pipe); отрывки, где все каналы тише −60 dBFS, пропускаются;
   - для каждой пары каналов считаются корреляция Пирсона и разница уровней;
   - `dual_mono`: корреляция ≥ 0,999 и |Δ| ≤ 0,5 дБ во всех голосовых отрывках;
   - `linked`: корреляция ≥ 0,98 при другом уровне — страховочная запись −6/−12 дБ (Zoom, Tascam, DJI Mic); основной канал — громкий без клиппинга;
   - `one_silent`: один канал тише −60 dBFS, в другом голос;
   - `independent`: всё остальное;
   - при > 2 каналах совпадающие каналы объединяются в группы (union-find).
   Анализ занимает секунды на файл и выполняется только для многоканального аудио.
2. `default_tracks` предлагает по одной дорожке на группу:
   - `dual_mono` и `linked` → `RecorderTrack(asset, (0, 1), name="моно: 2 одинаковых канала")`;
   - `one_silent` → дорожка активного канала, тихий канал выключен.
   Ручные дорожки (`refresh_tracks`) важнее; `uvf tracks` разделяет каналы обратно.
3. Общий помощник `effective_audio(asset)` возвращает каналы и `pan`. Им пользуются `selected_media`, рендер голоса (`channels=1` для моно-группы), `speakers.decode_channels`, звук `program` и вход Whisper:
   - `dual_mono` — `pan=mono|c0=c0`;
   - `linked` — основной канал;
   - `one_silent` — активный канал вместо `-ac 1`, которое занижает уровень на 6 дБ.
   Экспорт берёт число каналов из готового WAV и сам становится моно.
4. Экран Material и `uvf tracks` показывают класс и цифры (корреляция, разница уровней).

**Настройка:** `scan.channel_analysis: auto|off`.

**Тесты:**
- одинаковые каналы → одна дорожка и моно-голос;
- копия −12 дБ → `linked`;
- независимый шум → две дорожки;
- тихий правый канал → одна активная дорожка;
- ручное разделение переживает пересканирование.

**Готово** — как в пункте, плюс `voice/dji-0909.wav` моно, а `speakers` не предлагает двух спикеров.

#### Р9. Дайджесты больших файлов

**Причина.** SHA-256 исходников считают отпечатки `transcribe`, `sync` и `roughcut` (`_sha256`), а затем ещё раз `WhisperEngine._key`. `build_plan` вычисляет отпечатки всех этапов, некоторых — дважды, а цикл `uvf run` перепланирует после каждого этапа (`cli/main.py:125`). Кроме того, `StageManifest.reusable` при каждом планировании заново хэширует все артефакты (`_artifact_matches`), включая голоса и `program.mp4`. На полном pos17v2 (≈38 ГБ исходников) один проход планировщика читает десятки гигабайт.

**Решение.**
1. `core/digest.py`: `file_digest(path) -> str`.
   - Кэш — `_studio/cache/digests.json` (атомарная запись со слиянием; потерянное обновление означает лишь лишний пересчёт) плюс память процесса.
   - Запись: `(dev, ino, size, mtime_ns, ctime_ns)` → sha256 и `sample` (хэш первых, средних и последних 64 КиБ).
2. Если `stat` и `sample` совпали, дайджест берётся из кэша, иначе пересчитывается.
   - Гарантия «содержимое изменилось при прежних размере и mtime» сохраняется: запись с восстановлением mtime меняет `ctime`, поэтому `test_transcribe_identity.py` остаётся зелёным.
   - Для ФС с грубыми метками времени действует правило гонки, как в git: запись не доверяется, если `mtime/ctime` не старше момента записи в кэш хотя бы на 2 с.
   - На Windows, где ctime — время создания, решают `ino` (file index) и `sample`.
   - Перенос папки или смена прав вызывают один пересчёт.
3. Через `file_digest` идут все `_sha256` исходников и `_artifact_matches`. Runner кладёт в кэш дайджесты новых артефактов при публикации (они уже считаются в `describe_artifact`), поэтому следующая проверка манифеста — только `stat`.
4. `build_plan` запоминает отпечатки в пределах одного вызова.
5. Принудительная проверка: `scan.verify_digests: cached|always` и `uvf run --rehash`.

**Тесты:**
- повторный `build_plan` на неизменном проекте не читает содержимое (счётчик байт подменённого хэшера = 0);
- изменение содержимого с восстановленным mtime → пересчёт;
- повреждённый артефакт того же размера и mtime → без reuse;
- параллельные записи не портят кэш.

**Готово:** за `uvf run` каждый исходник хэшируется не больше одного раза; повторный `uvf plan` без изменений не читает исходники и укладывается в секунды.

#### Порядок работ и повторная приёмка

1. Р9 и Р7 — небольшие: ускоряют все следующие прогоны и чинят GUI-рендер субтитров.
2. Р1 — даёт измерение, по которому принимаются Р2 и Р3.
3. «А» → Р2 → Р3.
4. Р6 (новая схема кэша транскриптов), затем Р5.
5. Р8.
6. Р4 — самая крупная часть: `locate`, плитки транскрипции, `coverage` у потребителей.

Новые настройки добавляются в dataclass'ы `core/settings.py` с проверкой, на экран настроек и в таблицу миграции.

В `tools/make_fixtures.py` добавляются сценарии:
- выпадение 88 мс со щелчком;
- двойной моно и копия −12 дБ;
- длинный рекордер с коротким клипом;
- шумная голова до первой фразы;
- шумовой всплеск за 2 мин до фразы (VAD-склейка).

Повторная приёмка на реальном материале: DJI_0909 с рекордером (первые 30 мин и полный файл) в режимах `auto`, `complex` и `self_check=repair`. Проверяются:
- `verify`: p90 ≤ 5 мс, ступенька в отчёте;
- голова 0–186 с среди предложенных резов;
- одна дорожка рекордера;
- время Whisper по рекордеру и число полных чтений исходников.

Этот фрагмент стоит сохранить как приватный регрессионный набор.

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
