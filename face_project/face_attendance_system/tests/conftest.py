"""
Test Configuration
------------------
Pytest fixtures for the Face Attendance System tests.
"""

import pytest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import create_app
from app.database import db as _db


@pytest.fixture(scope="session")
def app():
    """Create application for testing."""
    app = create_app("testing")
    return app


@pytest.fixture(scope="function")
def db(app):
    """Create a fresh database for each test."""
    with app.app_context():
        _db.create_all()
        yield _db
        _db.session.rollback()
        _db.drop_all()


@pytest.fixture(scope="function")
def client(app, db):
    """Create a test client."""
    return app.test_client()


@pytest.fixture
def sample_user_data():
    """Sample user data for tests."""
    return {
        "employee_id": "EMP001",
        "first_name": "John",
        "last_name": "Doe",
        "email": "john.doe@test.com",
        "department": "Engineering",
        "role": "employee",
    }


@pytest.fixture
def sample_user(app, db, sample_user_data):
    """Create a sample user in the database."""
    from app.services.user_service import UserService
    with app.app_context():
        user, msg = UserService.create_user(sample_user_data)
        return user
