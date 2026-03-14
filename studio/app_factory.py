from __future__ import annotations

from flask import Flask

from studio.config import BASE_DIR, configure_app, ensure_project_folders
from studio.labels import APP_NAME
from studio.routes.data_routes import register_data_routes
from studio.routes.generative_routes import register_generative_routes
from studio.routes.media_routes import register_media_routes
from studio.routes.pages import register_page_routes
from studio.security import init_csrf


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )

    generated_secret = configure_app(app)
    ensure_project_folders(app)
    init_csrf(app)

    if generated_secret:
        app.logger.warning("FLASK_SECRET_KEY is not set; using an ephemeral secret key for this process.")

    @app.context_processor
    def inject_project_name():
