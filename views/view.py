"""
User View
"""

from flask import Blueprint, request, jsonify, render_template, redirect

view = Blueprint('view', __name__)


@view.route('/')
def index():
    return redirect("/")


@view.route('/status')
def status():
    return render_template('status.html')
