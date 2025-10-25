from flask import Flask
from flask_cors import CORS
from .ws import sock


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    # Initialize WebSocket support
    sock.init_app(app)

    from .routes.health import bp as health_bp
    from .routes.ingest import bp as ingest_bp
    from .routes.query import bp as query_bp
    from .routes.voice import bp as voice_bp
    from .routes.sessions import bp as sessions_bp
    from .routes.ws_sessions import register_ws_routes

    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(ingest_bp, url_prefix="/api")
    app.register_blueprint(query_bp, url_prefix="/api")
    app.register_blueprint(voice_bp, url_prefix="/api")
    app.register_blueprint(sessions_bp, url_prefix="/api")
    # Register WebSocket routes
    register_ws_routes()
    return app


