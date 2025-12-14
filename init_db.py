#!/usr/bin/env python3
"""Initialize database if it doesn't exist"""
from app import db
from run import app

with app.app_context():
    db.create_all()
    print("Database initialized successfully!")
