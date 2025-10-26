from flask import Flask, request
from flask_cors import CORS


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    from .routes.health import bp as health_bp
    from .routes.ingest import bp as ingest_bp
    from .routes.query import bp as query_bp
    from .routes.voice import bp as voice_bp
    from .routes.sessions import bp as sessions_bp
    from .routes.stream_sessions import bp as stream_sessions_bp
    from .routes.settings import bp as settings_bp

    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(ingest_bp, url_prefix="/api")
    app.register_blueprint(query_bp, url_prefix="/api")
    app.register_blueprint(voice_bp, url_prefix="/api")
    app.register_blueprint(sessions_bp, url_prefix="/api")
    app.register_blueprint(stream_sessions_bp, url_prefix="/api")
    app.register_blueprint(settings_bp, url_prefix="/api")

    # # Concise single-line HTTP logging
    # @app.before_request
    # def _log_request():
    #     try:
    #         raw = request.get_data(cache=True) or b""
    #         txt = (raw.decode(errors="replace") if raw else "")
    #         preview = (txt[:80] + "…") if len(txt) > 80 else txt
    #         print(f"[HTTP] {request.method} {request.path} req='{preview}' len={len(raw)}")
    #     except Exception as e:
    #         print(f"[HTTP] {request.method} {request.path} req=<error {e}>")

    # @app.after_request
    # def _log_response(response):
    #     try:
    #         data = response.get_data(as_text=True) or ""
    #         preview = (data[:80] + "…") if len(data) > 80 else data
    #         print(f"[HTTP] {request.method} {request.path} {response.status_code} res='{preview}' len={len(data)}")
    #     except Exception as e:
    #         print(f"[HTTP] {request.method} {request.path} {response.status_code} res=<error {e}>")
    #     return response
    return app


