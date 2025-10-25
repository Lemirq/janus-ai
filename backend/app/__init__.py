from flask import Flask
from flask_cors import CORS


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    from .routes.health import bp as health_bp
    from .routes.ingest import bp as ingest_bp
    from .routes.query import bp as query_bp
    from .routes.voice import bp as voice_bp

    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(ingest_bp, url_prefix="/api")
    app.register_blueprint(query_bp, url_prefix="/api")
    app.register_blueprint(voice_bp, url_prefix="/api")
    return app


