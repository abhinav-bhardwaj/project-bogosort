from flask import Flask
import os

from app.db import init_db


def create_app(config_name=None):
    app = Flask(__name__, instance_relative_config=True)
    
    # Load environment-based config
    if config_name is None:
        config_name = os.environ.get("FLASK_ENV", "development")
    
    if config_name == "development":
        from app.config import DevelopmentConfig
        app.config.from_object(DevelopmentConfig)
    elif config_name == "testing":
        from app.config import TestingConfig
        app.config.from_object(TestingConfig)
    elif config_name == "production":
        from app.config import ProductionConfig
        app.config.from_object(ProductionConfig)
    else:
        from app.config import Config
        app.config.from_object(Config)

    # Initialize DB
    init_db(app.config.get("SQL_URI"))

    # Load EDA cache at startup (optional, warnings if not available)
    try:
        from app.services.eda_service import load_eda_cache
        load_eda_cache('./analysis_and_inference/EDA/eda_cache.json')
    except FileNotFoundError:
        import logging
        logging.getLogger(__name__).warning(
            "EDA cache not found. Run compute_eda_cache.py to generate it."
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to load EDA cache: {e}")

    # Register blueprints
    from app.routes.main import main
    from app.routes.api import api
    from app.routes.dashboard import dashboard
    from app.routes.bogosort import bogosort_demo
    from app.routes.eda import eda

    app.register_blueprint(main)
    app.register_blueprint(api, url_prefix='/api')
    app.register_blueprint(dashboard, url_prefix='/dashboard')
    app.register_blueprint(bogosort_demo, url_prefix='/bogosort')
    app.register_blueprint(eda)

    # Register error handlers
    @app.errorhandler(404)
    def not_found(error):
        from flask import render_template
        return render_template('error.html',
                             status_code=404,
                             error_message='Page Not Found',
                             error_description='The page you are looking for does not exist.'), 404

    @app.errorhandler(500)
    def internal_error(error):
        from flask import render_template
        return render_template('error.html',
                             status_code=500,
                             error_message='Internal Server Error',
                             error_description='Something went wrong on our end. Please try again later.'), 500

    return app