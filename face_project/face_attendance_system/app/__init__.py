"""
Application Factory
-------------------
Creates and configures the Flask application instance.
Registers blueprints, initializes extensions, sets up logging and error handling.
"""

import logging
from flask import Flask

from config.settings import get_config
from app.database import db, init_db


def create_app(config_name=None):
    """
    Application factory pattern.

    Args:
        config_name: Configuration name ('development', 'production', 'testing')

    Returns:
        Configured Flask application instance
    """
    app = Flask(__name__)

    # Load configuration
    if config_name:
        from config.settings import config_map
        app.config.from_object(config_map.get(config_name, get_config()))
    else:
        app.config.from_object(get_config())

    # Initialize extensions
    db.init_app(app)

    # Setup logging
    _setup_logging(app)

    # Register blueprints
    _register_blueprints(app)

    # Create database tables
    with app.app_context():
        db.create_all()

    # Register error handlers
    from app.utils.error_handlers import register_error_handlers
    register_error_handlers(app)

    # Initialize security features
    from app.utils.security import init_security
    init_security(app)

    # Log application start
    from app.utils.logger import log_app_start
    log_app_start(config_name or "development")

    app.logger.info("Face Recognition Attendance System initialized")
    return app


def _register_blueprints(app):
    """Register all Flask blueprints."""
    from app.routes.main import main_bp
    from app.routes.user_routes import user_bp
    from app.routes.attendance_routes import attendance_bp
    from app.routes.api_routes import api_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(user_bp, url_prefix="/users")
    app.register_blueprint(attendance_bp, url_prefix="/attendance")
    app.register_blueprint(api_bp, url_prefix="/api")


def _setup_logging(app):
    """Configure application logging."""
    log_level = getattr(logging, app.config.get("LOG_LEVEL", "INFO"))
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
