from flask import Blueprint, render_template, jsonify, current_app
import json
import logging

logger = logging.getLogger(__name__)

eda = Blueprint('eda', __name__, url_prefix='/eda')


def _get_eda_data():
    """Load EDA data from cache file or return None if not available."""
    try:
        from app.services.eda_service import load_eda_cache, get_eda_data

        if not hasattr(current_app, '_eda_cache_loaded'):
            try:
                load_eda_cache('./analysis_and_inference/EDA/eda_cache.json')
                current_app._eda_cache_loaded = True
            except FileNotFoundError:
                logger.warning("EDA cache not found at ./analysis_and_inference/EDA/eda_cache.json")
                current_app._eda_cache_loaded = False
                return None

        if not current_app._eda_cache_loaded:
            return None

        return get_eda_data()
    except Exception as e:
        logger.error(f"Error loading EDA data: {e}")
        return None


@eda.route('/', methods=['GET'])
def dashboard():
    """Render EDA dashboard with pre-computed statistics."""
    data = _get_eda_data()

    if data is None:
        return render_template(
            'eda.html',
            eda_json='null',
            error_message='EDA cache not loaded. Run eda_processor.py first.'
        ), 503

    return render_template('eda.html', eda_json=json.dumps(data))


@eda.route('/api/data', methods=['GET'])
def api_data():
    """API endpoint to fetch EDA data as JSON."""
    data = _get_eda_data()

    if data is None:
        return jsonify({'error': 'EDA cache not loaded'}), 503

    return jsonify(data)


@eda.route('/api/overview', methods=['GET'])
def api_overview():
    """API endpoint for overview statistics."""
    data = _get_eda_data()

    if data is None:
        return jsonify({'error': 'EDA cache not loaded'}), 503

    return jsonify(data.get('overview', {}))


@eda.route('/api/top-features', methods=['GET'])
def api_top_features():
    """API endpoint for top predictive features."""
    data = _get_eda_data()

    if data is None:
        return jsonify({'error': 'EDA cache not loaded'}), 503

    return jsonify(data.get('top_features', []))
