#!/usr/bin/env python3
"""
G4F Auto-Provider Selector & API Server
=========================================
Script yang secara otomatis:
1. Scan semua provider g4f
2. Filter yang kompatibel dengan model pilihan user
3. Coba request dengan fallback otomatis
4. Return response ke user

Cocok untuk: Termux + proot-distro + PicoClaw
"""

import asyncio
import json
import time
import socket
import sys
from typing import Optional
from dataclasses import dataclass, field

# ============================================
# DEPENDENCY CHECK
# ============================================

def check_dependencies():
    """Pastikan semua dependency terinstall."""
    missing = []
    try:
        import g4f
    except ImportError:
        missing.append("g4f")
    try:
        from aiohttp import web
    except ImportError:
        missing.append("aiohttp")

    if missing:
        print("❌ Dependency belum terinstall:")
        for m in missing:
            print(f"   pip install {m}")
        sys.exit(1)

check_dependencies()

import g4f
from g4f.Provider import __all__ as all_provider_names
import g4f.Provider as ProviderModule
from g4f.providers.types import ProviderType
from aiohttp import web

# ============================================
# KONFIGURASI
# ============================================

@dataclass
class ServerConfig:
    """Konfigurasi server."""
    host: str = "127.0.0.1"        # Gunakan IP, bukan "localhost" (avoid DNS delay)
    port: int = 5000
    request_timeout: int = 60       # Timeout per provider (detik)
    max_retries: int = 3            # Maksimal provider yang dicoba
    debug: bool = True              # Tampilkan log detail
    excluded_providers: list = field(default_factory=lambda: [
        # Provider yang diketahui bermasalah atau butuh auth
        "PuterJS",
        "OpenaiAccount",
        "Gemini",
        "GeminiPro",
        "MetaAI",
        "CopilotAccount",
        "HuggingFace",
    ])

CONFIG = ServerConfig()

# ============================================
# PROVIDER SCANNER
# ============================================

class ProviderScanner:
    """
    Scan dan filter provider g4f yang tersedia.
    Membangun database provider yang kompatibel per model.
    """

    def __init__(self, config: ServerConfig):
        self.config = config
        self._cache: dict[str, list] = {}  # Cache hasil scan per model

    def get_all_providers(self) -> list[dict]:
        """
        Ambil SEMUA provider dari g4f library.
        Return list of dict dengan info setiap provider.
        """
        providers = []

        for name in sorted(all_provider_names):
            try:
                provider = getattr(ProviderModule, name, None)
                if provider is None:
                    continue

                # Kumpulkan informasi provider
                info = {
                    "name": name,
                    "provider": provider,
                    "needs_auth": getattr(provider, "needs_auth", False),
                    "working": getattr(provider, "working", False),
                    "supports_stream": getattr(provider, "supports_stream", False),
                    "models": self._get_provider_models(provider),
                    "url": getattr(provider, "url", ""),
                }
                providers.append(info)

            except Exception as e:
                if self.config.debug:
                    print(f"  ⚠️  Skip {name}: {e}")

        return providers

    def _get_provider_models(self, provider) -> list[str]:
        """Ekstrak daftar model yang didukung provider."""
        models = []

        # Coba berbagai cara mendapatkan list model
        if hasattr(provider, "models") and provider.models:
            if isinstance(provider.models, (list, tuple)):
                models = list(provider.models)
            elif isinstance(provider.models, dict):
                models = list(provider.models.keys())

        if hasattr(provider, "model_aliases") and provider.model_aliases:
            if isinstance(provider.model_aliases, dict):
                models.extend(provider.model_aliases.keys())
                models.extend(provider.model_aliases.values())

        if hasattr(provider, "default_model") and provider.default_model:
            models.append(provider.default_model)

        if hasattr(provider, "supported_models") and provider.supported_models:
            models.extend(provider.supported_models)

        # Deduplicate dan bersihkan
        return list(set(str(m) for m in models if m))

    def find_compatible(self, model: str) -> list[dict]:
        """
        Cari provider yang kompatibel dengan model tertentu.

        Args:
            model: Nama model AI (contoh: "gpt-4o-mini")

        Returns:
            List provider yang kompatibel, sudah diurutkan prioritas.
        """
        # Cek cache
        if model in self._cache:
            if self.config.debug:
                print(f"  📦 Cache hit untuk model: {model}")
            return self._cache[model]

        all_providers = self.get_all_providers()
        compatible = []

        for info in all_providers:
            name = info["name"]

            # ── Filter 1: Skip provider yang di-exclude
            if name in self.config.excluded_providers:
                if self.config.debug:
                    print(f"  ⛔ {name}: excluded")
                continue

            # ── Filter 2: Skip yang butuh autentikasi
            if info["needs_auth"]:
                if self.config.debug:
                    print(f"  🔐 {name}: needs auth, skip")
                continue

            # ── Filter 3: Skip yang tidak working
            if not info["working"]:
                if self.config.debug:
                    print(f"  💀 {name}: not working, skip")
                continue

            # ── Filter 4: Cek kompatibilitas model
            provider_models = info["models"]

            model_compatible = False

            if not provider_models:
                # Provider tanpa daftar model = mungkin support semua
                model_compatible = True
                info["match_type"] = "universal"
            elif model in provider_models:
                # Match exact
                model_compatible = True
                info["match_type"] = "exact"
            else:
                # Match partial (misal "gpt-4o" match "gpt-4o-mini")
                model_lower = model.lower()
                for pm in provider_models:
                    if model_lower in str(pm).lower() or str(pm).lower() in model_lower:
                        model_compatible = True
                        info["match_type"] = "partial"
                        break

            if model_compatible:
                compatible.append(info)
                if self.config.debug:
                    print(f"  ✅ {name}: kompatibel ({info.get('match_type', '?')})")
            else:
                if self.config.debug:
                    print(f"  ❌ {name}: model '{model}' tidak didukung")

        # ── Urutkan prioritas: exact > partial > universal
        priority = {"exact": 0, "partial": 1, "universal": 2}
        compatible.sort(key=lambda x: priority.get(x.get("match_type", ""), 99))

        # Simpan ke cache
        self._cache[model] = compatible

        return compatible

    def clear_cache(self):
        """Bersihkan cache."""
        self._cache.clear()


# ============================================
# REQUEST HANDLER
# ============================================

class G4FRequestHandler:
    """
    Handle request ke g4f dengan auto-retry
    menggunakan provider yang berbeda.
    """

    def __init__(self, scanner: ProviderScanner, config: ServerConfig):
        self.scanner = scanner
        self.config = config

    async def process_request(
        self,
        model: str,
        messages: list[dict],
        stream: bool = False,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        preferred_provider: Optional[str] = None,
    ) -> dict:
        """
        Proses request chat completion.

        1. Cari provider kompatibel
        2. Coba satu per satu dengan fallback
        3. Return response pertama yang berhasil
        """
        start_time = time.time()

        # ── Cari provider kompatibel
        print(f"\n{'='*50}")
        print(f"📨 Request: model={model}, messages={len(messages)}")
        print(f"{'='*50}")

        # Kalau user specify provider tertentu
        if preferred_provider:
            compatible = [
                p for p in self.scanner.find_compatible(model)
                if p["name"] == preferred_provider
            ]
            if not compatible:
                # Tetap coba provider yang diminta
                try:
                    prov = getattr(ProviderModule, preferred_provider)
                    compatible = [{
                        "name": preferred_provider,
                        "provider": prov,
                        "match_type": "forced",
                    }]
                except AttributeError:
                    return self._error_response(
                        f"Provider '{preferred_provider}' tidak ditemukan"
                    )
        else:
            compatible = self.scanner.find_compatible(model)

        if not compatible:
            return self._error_response(
                f"Tidak ada provider yang kompatibel dengan model '{model}'"
            )

        print(f"\n🔍 Ditemukan {len(compatible)} provider kompatibel:")
        for i, p in enumerate(compatible):
            print(f"   {i+1}. {p['name']} ({p.get('match_type', '?')})")

        # ── Coba provider satu per satu
        errors = []
        max_tries = min(len(compatible), self.config.max_retries)

        for i, provider_info in enumerate(compatible[:max_tries]):
            provider_name = provider_info["name"]
            provider = provider_info["provider"]

            print(f"\n🔄 Mencoba [{i+1}/{max_tries}]: {provider_name}...")

            try:
                response_text = await self._call_provider(
                    provider=provider,
                    model=model,
                    messages=messages,
                    stream=stream,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                elapsed = time.time() - start_time
                print(f"✅ Berhasil via {provider_name} ({elapsed:.2f}s)")

                return self._success_response(
                    content=response_text,
                    model=model,
                    provider=provider_name,
                    elapsed=elapsed,
                )

            except Exception as e:
                error_msg = f"{provider_name}: {type(e).__name__}: {str(e)[:200]}"
                errors.append(error_msg)
                print(f"❌ Gagal: {error_msg}")
                continue

        # ── Semua provider gagal
        elapsed = time.time() - start_time
        return self._error_response(
            f"Semua provider gagal setelah {max_tries} percobaan ({elapsed:.2f}s)",
            errors=errors,
        )

    async def _call_provider(
        self,
        provider,
        model: str,
        messages: list[dict],
        stream: bool,
        temperature: float,
        max_tokens: Optional[int],
    ) -> str:
        """
        Panggil g4f dengan provider spesifik.
        Menggunakan asyncio dengan timeout.
        """
        response_text = ""

        # Build kwargs
        kwargs = {
            "model": model,
            "messages": messages,
            "provider": provider,
            "stream": False,  # Non-stream untuk simplicity
        }

        if temperature != 1.0:
            kwargs["temperature"] = temperature
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        # Panggil g4f dengan timeout
        result = await asyncio.wait_for(
            g4f.ChatCompletion.create_async(**kwargs),
            timeout=self.config.request_timeout,
        )

        if isinstance(result, str):
            response_text = result
        elif hasattr(result, '__aiter__'):
            # Async generator
            async for chunk in result:
                if isinstance(chunk, str):
                    response_text += chunk
                elif hasattr(chunk, 'content'):
                    response_text += chunk.content or ""
        else:
            response_text = str(result)

        if not response_text.strip():
            raise ValueError("Response kosong dari provider")

        return response_text

    def _success_response(
        self, content: str, model: str, provider: str, elapsed: float
    ) -> dict:
        """Format response sukses (OpenAI-compatible)."""
        return {
            "id": f"chatcmpl-g4f-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "provider": provider,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "elapsed_seconds": round(elapsed, 2),
        }

    def _error_response(self, message: str, errors: list = None) -> dict:
        """Format response error."""
        resp = {
            "error": {
                "message": message,
                "type": "provider_error",
                "code": 503,
            }
        }
        if errors:
            resp["error"]["details"] = errors
        return resp


# ============================================
# API SERVER (OpenAI-Compatible)
# ============================================

class APIServer:
    """
    HTTP Server yang expose endpoint OpenAI-compatible.
    Bisa langsung dipakai oleh PicoClaw atau client lain.
    """

    def __init__(self, config: ServerConfig):
        self.config = config
        self.scanner = ProviderScanner(config)
        self.handler = G4FRequestHandler(self.scanner, config)
        self.app = web.Application()
        self._setup_routes()

    def _setup_routes(self):
        """Daftarkan semua endpoint."""
        self.app.router.add_get("/", self._handle_root)
        self.app.router.add_get("/v1/models", self._handle_models)
        self.app.router.add_post("/v1/chat/completions", self._handle_chat)
        self.app.router.add_get("/v1/providers", self._handle_providers)
        self.app.router.add_get("/v1/providers/{model}", self._handle_providers_for_model)
        self.app.router.add_get("/health", self._handle_health)

    # ── Root endpoint
    async def _handle_root(self, request: web.Request) -> web.Response:
        """Landing page."""
        info = {
            "service": "G4F Auto-Provider API",
            "version": "1.0.0",
            "endpoints": {
                "GET  /v1/models": "Daftar model tersedia",
                "POST /v1/chat/completions": "Chat completion (OpenAI-compatible)",
                "GET  /v1/providers": "Semua provider tersedia",
                "GET  /v1/providers/{model}": "Provider kompatibel untuk model",
                "GET  /health": "Health check",
            },
            "config_for_picoclaw": {
                "model_name": "g4f",
                "model": "gpt-4o-mini",
                "api_base": f"http://{self.config.host}:{self.config.port}/v1",
                "api_key": "any-value-works",
            },
        }
        return web.json_response(info, dumps=lambda x: json.dumps(x, indent=2))

    # ── List models
    async def _handle_models(self, request: web.Request) -> web.Response:
        """Return daftar model yang tersedia."""
        all_models = set()
        providers = self.scanner.get_all_providers()

        for p in providers:
            if p["name"] not in self.config.excluded_providers and not p["needs_auth"]:
                all_models.update(p["models"])

        model_list = {
            "object": "list",
            "data": [
                {
                    "id": m,
                    "object": "model",
                    "owned_by": "g4f",
                }
                for m in sorted(all_models)
                if m  # skip empty
            ],
        }
        return web.json_response(model_list)

    # ── Chat completions (ENDPOINT UTAMA)
    async def _handle_chat(self, request: web.Request) -> web.Response:
        """
        Handle chat completion request.
        Format: OpenAI-compatible.
        """
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response(
                {"error": {"message": "Invalid JSON body", "code": 400}},
                status=400,
            )

        # Parse request body
        model = body.get("model", "gpt-4o-mini")
        messages = body.get("messages", [])
        stream = body.get("stream", False)
        temperature = body.get("temperature", 1.0)
        max_tokens = body.get("max_tokens", None)
        provider = body.get("provider", None)  # Optional: paksa provider tertentu

        # Validasi messages
        if not messages:
            return web.json_response(
                {"error": {"message": "messages tidak boleh kosong", "code": 400}},
                status=400,
            )

        # Proses request
        result = await self.handler.process_request(
            model=model,
            messages=messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            preferred_provider=provider,
        )

        # Cek apakah error
        if "error" in result:
            return web.json_response(result, status=result["error"].get("code", 503))

        return web.json_response(result)

    # ── List providers
    async def _handle_providers(self, request: web.Request) -> web.Response:
        """Return semua provider dan statusnya."""
        providers = self.scanner.get_all_providers()
        data = []
        for p in providers:
            data.append({
                "name": p["name"],
                "working": p["working"],
                "needs_auth": p["needs_auth"],
                "supports_stream": p["supports_stream"],
                "models_count": len(p["models"]),
                "models": p["models"][:10],  # Limit 10 untuk readability
                "excluded": p["name"] in self.config.excluded_providers,
                "url": p["url"],
            })
        return web.json_response(
            {"providers": data, "total": len(data)},
            dumps=lambda x: json.dumps(x, indent=2),
        )

    # ── Providers untuk model spesifik
    async def _handle_providers_for_model(self, request: web.Request) -> web.Response:
        """Return provider yang kompatibel untuk model tertentu."""
        model = request.match_info["model"]
        compatible = self.scanner.find_compatible(model)

        data = [
            {
                "name": p["name"],
                "match_type": p.get("match_type", "unknown"),
                "supports_stream": p.get("supports_stream", False),
            }
            for p in compatible
        ]

        return web.json_response({
            "model": model,
            "compatible_providers": data,
            "count": len(data),
        })

    # ── Health check
    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            "status": "ok",
            "timestamp": int(time.time()),
        })

    def run(self):
        """Jalankan server."""
        print(f"""
╔══════════════════════════════════════════════════╗
║        G4F Auto-Provider API Server              ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  🌐 Server: http://{self.config.host}:{self.config.port}          ║
║  📡 API:    http://{self.config.host}:{self.config.port}/v1       ║
║                                                  ║
║  Endpoints:                                      ║
║    GET  /v1/models              - List models    ║
║    POST /v1/chat/completions    - Chat API       ║
║    GET  /v1/providers           - All providers  ║
║    GET  /v1/providers/{{model}}   - Per model     ║
║                                                  ║
║  PicoClaw Config:                                ║
║  {{                                               ║
║    "api_base": "http://{self.config.host}:{self.config.port}/v1", ║
║    "api_key": "any-value",                       ║
║    "model": "gpt-4o-mini"                        ║
║  }}                                               ║
║                                                  ║
╚══════════════════════════════════════════════════╝
        """)

        # Force IPv4 socket
        original_getaddrinfo = socket.getaddrinfo
        def ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            return original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
        socket.getaddrinfo = ipv4_getaddrinfo

        web.run_app(
            self.app,
            host=self.config.host,
            port=self.config.port,
            print=None,  # Suppress default print (kita sudah print di atas)
        )


# ============================================
# CLI MODE (Tanpa server, langsung di terminal)
# ============================================

async def cli_mode():
    """
    Mode interaktif di terminal.
    Untuk testing tanpa perlu jalankan server.
    """
    config = ServerConfig()
    scanner = ProviderScanner(config)
    handler = G4FRequestHandler(scanner, config)

    print("╔══════════════════════════════════════╗")
    print("║   G4F CLI Mode (Interactive Chat)    ║")
    print("║   Ketik 'quit' untuk keluar          ║")
    print("║   Ketik 'model:xxx' ganti model      ║")
    print("║   Ketik 'providers' lihat provider    ║")
    print("╚══════════════════════════════════════╝")

    model = "gpt-4o-mini"
    messages = []

    while True:
        try:
            user_input = input(f"\n[{model}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("👋 Bye!")
            break

        if user_input.lower() == "providers":
            compatible = scanner.find_compatible(model)
            print(f"\nProvider kompatibel untuk '{model}':")
            for i, p in enumerate(compatible):
                print(f"  {i+1}. {p['name']} ({p.get('match_type', '?')})")
            continue

        if user_input.lower().startswith("model:"):
            model = user_input.split(":", 1)[1].strip()
            scanner.clear_cache()
            messages = []
            print(f"✅ Model diganti ke: {model}")
            continue

        if user_input.lower() == "clear":
            messages = []
            print("✅ Conversation cleared")
            continue

        # Tambah ke history
        messages.append({"role": "user", "content": user_input})

        # Process
        result = await handler.process_request(
            model=model,
            messages=messages,
        )

        if "error" in result:
            print(f"\n❌ Error: {result['error']['message']}")
            if "details" in result["error"]:
                for d in result["error"]["details"]:
                    print(f"   - {d}")
            messages.pop()  # Hapus pesan user yang gagal
        else:
            content = result["choices"][0]["message"]["content"]
            provider = result.get("provider", "?")
            elapsed = result.get("elapsed_seconds", 0)
            print(f"\n🤖 [{provider} | {elapsed}s] Assistant: {content}")
            messages.append({"role": "assistant", "content": content})


# ============================================
# ENTRY POINT
# ============================================

def main():
    """
    Entry point utama.

    Usage:
        python g4f_server.py              → Jalankan API server
        python g4f_server.py --cli        → Mode interactive chat
        python g4f_server.py --scan MODEL → Scan provider untuk model
        python g4f_server.py --port 8080  → Custom port
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="G4F Auto-Provider API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python g4f_server.py                    # Server di port 5000
  python g4f_server.py --port 8080        # Server di port 8080
  python g4f_server.py --cli              # Chat interaktif
  python g4f_server.py --scan gpt-4o-mini # Cari provider
        """,
    )

    parser.add_argument(
        "--cli", action="store_true",
        help="Jalankan mode interactive chat (tanpa server)",
    )
    parser.add_argument(
        "--scan", type=str, metavar="MODEL",
        help="Scan provider kompatibel untuk model tertentu",
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Port server (default: 5000)",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Host server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--timeout", type=int, default=60,
        help="Timeout per provider dalam detik (default: 60)",
    )
    parser.add_argument(
        "--retries", type=int, default=3,
        help="Maksimal provider yang dicoba (default: 3)",
    )
    parser.add_argument(
        "--no-debug", action="store_true",
        help="Nonaktifkan debug logging",
    )

    args = parser.parse_args()

    # Build config
    config = ServerConfig(
        host=args.host,
        port=args.port,
        request_timeout=args.timeout,
        max_retries=args.retries,
        debug=not args.no_debug,
    )

    # ── Mode: Scan provider
    if args.scan:
        print(f"\n🔍 Scanning provider untuk model: {args.scan}\n")
        scanner = ProviderScanner(config)
        compatible = scanner.find_compatible(args.scan)

        print(f"\n{'='*50}")
        print(f"📊 Hasil: {len(compatible)} provider kompatibel")
        print(f"{'='*50}")

        for i, p in enumerate(compatible):
            print(f"\n  [{i+1}] {p['name']}")
            print(f"      Match: {p.get('match_type', '?')}")
            print(f"      Stream: {p.get('supports_stream', '?')}")
            print(f"      Models: {', '.join(p['models'][:5])}")
        return

    # ── Mode: CLI interaktif
    if args.cli:
        asyncio.run(cli_mode())
        return

    # ── Mode: API Server (default)
    server = APIServer(config)
    server.run()


if __name__ == "__main__":
    main()
