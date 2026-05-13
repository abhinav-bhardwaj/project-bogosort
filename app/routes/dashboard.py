"""
dashboard.py — routes for dashboard views

This module keeps dashboard routing intentionally minimal by separating view
rendering from application logic. The Blueprint acts only as a navigation layer
between URLs and templates.

Routes are split into standard and “nerdy” dashboard variants so alternative
interfaces can evolve independently without changing shared navigation logic.

"""

from flask import Blueprint, render_template

#dashboard = Blueprint('dashboard', __name__,url_prefix="/dashboard")
dashboard = Blueprint('dashboard', __name__)


#@dashboard.route('/')
#def dashboard_page():
#    return render_template('dashboard.html')

@dashboard.route("/nerdy/")
def nerdy_dashboard():
    return render_template("dashboard_nerdy.html")