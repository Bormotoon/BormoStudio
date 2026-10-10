"""Pinned Whisper and GGUF downloads with per-file integrity data.

Each entry fixes a Hugging Face commit. Large files carry the SHA256 that Hugging
Face publishes for their LFS objects; small files carry their git blob SHA1.
Both were read from the pinned commit, so a download either reproduces those
exact bytes or is rejected.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RemoteFile:
    path: str
    size: int
    algorithm: str  # "sha256" or "gitsha1"
    digest: str


@dataclass(frozen=True, slots=True)
class CatalogModel:
    name: str
    kind: str  # "whisper" or "gguf"
    title: str
    repo: str
    revision: str
    files: tuple[RemoteFile, ...]
    whisper_name: str = ""

    @property
    def size(self) -> int:
        return sum(item.size for item in self.files)

    def url(self, item: RemoteFile) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{item.path}"


def _files(*items: tuple[str, int, str, str]) -> tuple[RemoteFile, ...]:
    return tuple(RemoteFile(*item) for item in items)


_TOKENIZER = ("tokenizer.json", 2203239, "gitsha1", "7818adb6de9fa3064d3ff81226fdd675be1f6344")
_VOCABULARY = ("vocabulary.txt", 459861, "gitsha1", "c9074644d9d1205686f16d411564729461324b75")
_PREPROCESSOR = (
    "preprocessor_config.json", 340, "gitsha1", "931c77a740890c46365c7ae0c9d350ba3cca908f"
)
_VOCABULARY_V3 = (
    "vocabulary.json", 1068114, "gitsha1", "0adcd01e7c237205d593b707e66dd5d7bc785d2d"
)

_MODELS = (
    CatalogModel(
        "whisper-tiny", "whisper", "Whisper tiny (75 MB, tests only)",
        "Systran/faster-whisper-tiny", "d90ca5fe260221311c53c58e660288d3deb8d356",
        _files(
            ("config.json", 2249, "gitsha1", "3baa18e2b321a2f489614607852a729fcd516480"),
            ("model.bin", 75538270, "sha256",
             "dcb76c6586fc06cbdac6dd21f14cfd129cc4cdd9dce19bf4ffa62e59cbe6e6d1"),
            _TOKENIZER, _VOCABULARY,
        ),
        "tiny",
    ),
    CatalogModel(
        "whisper-base", "whisper", "Whisper base (145 MB)",
        "Systran/faster-whisper-base", "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66",
        _files(
            ("config.json", 2309, "gitsha1", "867cf1a0fece1394e01d55e287ba2f09a577c046"),
            ("model.bin", 145217532, "sha256",
             "d01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9"),
            _TOKENIZER, _VOCABULARY,
        ),
        "base",
    ),
    CatalogModel(
        "whisper-small", "whisper", "Whisper small (484 MB)",
        "Systran/faster-whisper-small", "536b0662742c02347bc0e980a01041f333bce120",
        _files(
            ("config.json", 2370, "gitsha1", "e5047537059bd8f182d9ca64c470201585015187"),
            ("model.bin", 483546902, "sha256",
             "3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671"),
            _TOKENIZER, _VOCABULARY,
        ),
        "small",
    ),
    CatalogModel(
        "whisper-medium", "whisper", "Whisper medium (1.5 GB)",
        "Systran/faster-whisper-medium", "08e178d48790749d25932bbc082711ddcfdfbc4f",
        _files(
            ("config.json", 2257, "gitsha1", "242aa06a480a7b5509375c645097e87af5136774"),
            ("model.bin", 1527906378, "sha256",
             "9b45e1009dcc4ab601eff815b61d80e60ce3fd8c74c1a14f4a282258286b51ae"),
            _TOKENIZER, _VOCABULARY,
        ),
        "medium",
    ),
    CatalogModel(
        "whisper-large-v3", "whisper", "Whisper large-v3 (3.1 GB, default)",
        "Systran/faster-whisper-large-v3", "edaa852ec7e145841d8ffdb056a99866b5f0a478",
        _files(
            ("config.json", 2394, "gitsha1", "75336feae814999bae6ccccdecf177639ffc6f9d"),
            ("model.bin", 3087284237, "sha256",
             "69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1"),
            _PREPROCESSOR,
            ("tokenizer.json", 2480617, "gitsha1", "3a5e2ba63acdcac9a19ba56cf9bd27f185bfff61"),
            _VOCABULARY_V3,
        ),
        "large-v3",
    ),
    CatalogModel(
        "whisper-large-v3-turbo", "whisper", "Whisper large-v3 turbo (1.6 GB, faster)",
        "deepdml/faster-whisper-large-v3-turbo-ct2", "4df90f75321148c3a29a9e2351b7ddf8f5b115a8",
        _files(
            ("config.json", 2263, "gitsha1", "0351d1d6870005e865747b781b5d7c23ea0459cd"),
            ("model.bin", 1617884929, "sha256",
             "e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da"),
            _PREPROCESSOR,
            ("tokenizer.json", 2710337, "gitsha1", "17456db595adc78a973f97d69d8cb50bc87c0b1c"),
            _VOCABULARY_V3,
        ),
        "large-v3-turbo",
    ),
    CatalogModel(
        "gguf-gemma-4-e4b-it-q4", "gguf", "Gemma 4 E4B instruct Q4_0 (4.6 GB)",
        "ggml-org/gemma-4-E4B-it-GGUF", "b8093469224f83f5c38f691eb906c380e9e63114",
        _files(("gemma-4-E4B-it-Q4_0.gguf", 4590807392, "sha256",
                "a555b900214b477d8880e7832e0b8925e139b0159640036b09fe472b6f2097f2")),
    ),
    CatalogModel(
        "gguf-gemma-4-e4b-it-q8", "gguf", "Gemma 4 E4B instruct Q8_0 (8.0 GB)",
        "ggml-org/gemma-4-E4B-it-GGUF", "b8093469224f83f5c38f691eb906c380e9e63114",
        _files(("gemma-4-E4B-it-Q8_0.gguf", 8031242688, "sha256",
                "34be82b17b4942d389b9b527170c4b058027abdd32531fda063d3d97dd8ce80a")),
    ),
    CatalogModel(
        "gguf-gemma-4-12b-it-q4", "gguf", "Gemma 4 12B instruct Q4_0 (7.2 GB)",
        "ggml-org/gemma-4-12B-it-GGUF", "e3e681731089efaa3f0917336944ac64752db8ba",
        _files(("gemma-4-12B-it-Q4_0.gguf", 7219673216, "sha256",
                "3712b9bd32cae83a22f67ee7a4466d8d7a4f21646ac8a07d19bf9418e8767a70")),
    ),
    CatalogModel(
        "gguf-gemma-4-26b-a4b-it-q4", "gguf", "Gemma 4 26B-A4B instruct Q4_0 (14.6 GB)",
        "ggml-org/gemma-4-26B-A4B-it-GGUF", "bb4531cda34d1ea09d9814959ed4d5833cf2a4c8",
        _files(("gemma-4-26B-A4B-it-Q4_0.gguf", 14618145824, "sha256",
                "d208665ab1cd3a69f7a9a4bc59430e8448c8093d9b06334f566ac59d6d504a03")),
    ),
)

CATALOG: dict[str, CatalogModel] = {model.name: model for model in _MODELS}


def whisper_entry(model: str) -> CatalogModel | None:
    """Catalog entry for a faster-whisper model name such as ``large-v3``."""
    return next(
        (item for item in _MODELS if item.kind == "whisper" and item.whisper_name == model),
        None,
    )
