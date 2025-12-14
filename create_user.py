#!/usr/bin/env python3
"""Create a default user for testing"""
from app import db
from app.models import User
from run import app
from werkzeug.security import generate_password_hash

with app.app_context():
    # Check if any users exist
    existing_users = User.query.all()
    print(f"Existing users: {len(existing_users)}")
    for user in existing_users:
        print(f"  - {user.username}")
    
    # Create admin user if it doesn't exist
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin_user = User(
            username='admin',
            password_hash=generate_password_hash('admin123')
        )
        db.session.add(admin_user)
        db.session.commit()
        print("Created admin user with password: admin123")
    else:
        print("Admin user already exists")
    
    # Create test user if it doesn't exist
    test = User.query.filter_by(username='test').first()
    if not test:
        test_user = User(
            username='test',
            password_hash=generate_password_hash('test123')
        )
        db.session.add(test_user)
        db.session.commit()
        print("Created test user with password: test123")
    else:
        print("Test user already exists")
