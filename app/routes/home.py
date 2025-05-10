from flask import Blueprint, render_template
from datetime import datetime

home_bp = Blueprint('home', __name__)

@home_bp.route("/")
def home():
    current_time = datetime.now().strftime("%I:%M:%S %p")
    return render_template("index.html", time=current_time)