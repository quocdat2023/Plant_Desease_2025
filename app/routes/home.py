from flask import Blueprint, render_template
from datetime import datetime

home_bp = Blueprint('home', __name__)

@home_bp.route("/")
def home():
    current_time = datetime.now().strftime("%I:%M:%S %p")
    return render_template("index.html", time=current_time)


@home_bp.route("/home")
def homes():
    current_time = datetime.now().strftime("%I:%M:%S %p")
    return render_template("index.html", time=current_time)

@home_bp.route("/plant_detection")
def plant_detection():
    current_time = datetime.now().strftime("%I:%M:%S %p")
    return render_template("plant_detection.html", time=current_time)


@home_bp.route("/plant_recommendation")
def plant_recommendation():
    current_time = datetime.now().strftime("%I:%M:%S %p")
    return render_template("plant_recommendation.html", time=current_time)


@home_bp.route("/plant_fertilizer")
def plant_fertilizer():
    current_time = datetime.now().strftime("%I:%M:%S %p")
    return render_template("plant_fertilizer.html", time=current_time)


@home_bp.route("/register")
def register():
    current_time = datetime.now().strftime("%I:%M:%S %p")
    return render_template("register.html", time=current_time)

@home_bp.route("/login")
def login():
    current_time = datetime.now().strftime("%I:%M:%S %p")
    return render_template("login.html", time=current_time)
