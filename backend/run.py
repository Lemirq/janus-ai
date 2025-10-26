import os

# Configure environment before importing app/tokenizers to avoid fork-related warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from app import create_app


def main() -> None:
    app = create_app()
    app.run(host="0.0.0.0", port=2025, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()


