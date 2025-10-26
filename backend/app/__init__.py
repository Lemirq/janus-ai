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

    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(ingest_bp, url_prefix="/api")
    app.register_blueprint(query_bp, url_prefix="/api")
    app.register_blueprint(voice_bp, url_prefix="/api")
    app.register_blueprint(sessions_bp, url_prefix="/api")
    app.register_blueprint(stream_sessions_bp, url_prefix="/api")

    # Verbose HTTP logging
    @app.before_request
    def _log_request():
        try:
            body = request.get_data(cache=True) or b""
            body_display = body.decode(errors="replace")
            if len(body_display) > 2000:
                body_display = body_display[:2000] + "...<truncated>"
            print(f"[HTTP REQ] {request.method} {request.path} headers={dict(request.headers)} body={body_display}")
        except Exception as e:
            print(f"[HTTP REQ] {request.method} {request.path} <body read error: {e}>")

    @app.after_request
    def _log_response(response):
        try:
            data = response.get_data(as_text=True)
            display = data if data is not None else ""
            if len(display) > 2000:
                display = display[:2000] + "...<truncated>"
            print(f"[HTTP RES] {request.method} {request.path} status={response.status_code} body={display}")
        except Exception as e:
            print(f"[HTTP RES] {request.method} {request.path} <response read error: {e}>")
        return response
    return app


