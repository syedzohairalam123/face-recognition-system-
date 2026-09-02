"""
UserService Tests
-----------------
Unit tests for user management operations.
"""

import pytest
from app.services.user_service import UserService


class TestCreateUser:
    """Tests for user creation."""

    def test_create_user_success(self, app, db):
        """Test successful user creation."""
        with app.app_context():
            data = {
                "employee_id": "EMP100",
                "first_name": "Jane",
                "last_name": "Smith",
                "email": "jane.smith@test.com",
                "department": "Marketing",
            }
            user, message = UserService.create_user(data)

            assert user is not None
            assert user.employee_id == "EMP100"
            assert user.full_name == "Jane Smith"
            assert user.email == "jane.smith@test.com"
            assert user.is_active is True
            assert user.face_registered is False
            assert "success" in message.lower()

    def test_create_user_duplicate_id(self, app, db, sample_user):
        """Test creating user with duplicate employee ID."""
        with app.app_context():
            data = {
                "employee_id": "EMP001",  # Same as sample_user
                "first_name": "Another",
                "last_name": "Person",
                "email": "another@test.com",
            }
            user, message = UserService.create_user(data)

            assert user is None
            assert "already exists" in message.lower()

    def test_create_user_duplicate_email(self, app, db, sample_user):
        """Test creating user with duplicate email."""
        with app.app_context():
            data = {
                "employee_id": "EMP002",
                "first_name": "Another",
                "last_name": "Person",
                "email": "john.doe@test.com",  # Same as sample_user
            }
            user, message = UserService.create_user(data)

            assert user is None
            assert "already registered" in message.lower()

    def test_create_user_missing_field(self, app, db):
        """Test creating user with missing required field."""
        with app.app_context():
            data = {
                "employee_id": "EMP101",
                # Missing first_name, last_name, email
            }
            user, message = UserService.create_user(data)

            assert user is None
            assert "missing" in message.lower()


class TestGetUser:
    """Tests for user retrieval."""

    def test_get_by_id(self, app, db, sample_user):
        """Test getting user by database ID."""
        with app.app_context():
            user = UserService.get_user_by_id(sample_user.id)
            assert user is not None
            assert user.employee_id == "EMP001"

    def test_get_by_employee_id(self, app, db, sample_user):
        """Test getting user by employee ID."""
        with app.app_context():
            user = UserService.get_user_by_employee_id("EMP001")
            assert user is not None
            assert user.full_name == "John Doe"

    def test_get_nonexistent_user(self, app, db):
        """Test getting a user that doesn't exist."""
        with app.app_context():
            user = UserService.get_user_by_id(9999)
            assert user is None

    def test_get_all_users(self, app, db, sample_user):
        """Test getting all users."""
        with app.app_context():
            users = UserService.get_all_users()
            assert len(users) >= 1


class TestUpdateUser:
    """Tests for user updates."""

    def test_update_user_success(self, app, db, sample_user):
        """Test successful user update."""
        with app.app_context():
            data = {"first_name": "Johnny", "department": "Sales"}
            user, message = UserService.update_user(sample_user.id, data)

            assert user is not None
            assert user.first_name == "Johnny"
            assert user.department == "Sales"

    def test_update_nonexistent_user(self, app, db):
        """Test updating a user that doesn't exist."""
        with app.app_context():
            data = {"first_name": "Ghost"}
            user, message = UserService.update_user(9999, data)

            assert user is None
            assert "not found" in message.lower()


class TestDeleteUser:
    """Tests for user deactivation."""

    def test_deactivate_user(self, app, db, sample_user):
        """Test soft-deleting a user."""
        with app.app_context():
            message = UserService.delete_user(sample_user.id)
            assert "success" in message.lower()

            user = UserService.get_user_by_id(sample_user.id)
            assert user.is_active is False

    def test_deactivate_nonexistent_user(self, app, db):
        """Test deactivating a user that doesn't exist."""
        with app.app_context():
            message = UserService.delete_user(9999)
            assert "not found" in message.lower()


class TestSearchUsers:
    """Tests for user search."""

    def test_search_by_name(self, app, db, sample_user):
        """Test searching users by name."""
        with app.app_context():
            results = UserService.search_users("John")
            assert len(results) >= 1
            assert any(u.employee_id == "EMP001" for u in results)

    def test_search_by_employee_id(self, app, db, sample_user):
        """Test searching users by employee ID."""
        with app.app_context():
            results = UserService.search_users("EMP001")
            assert len(results) >= 1

    def test_search_no_results(self, app, db, sample_user):
        """Test search with no matching results."""
        with app.app_context():
            results = UserService.search_users("ZZZZZ")
            assert len(results) == 0


class TestUserStats:
    """Tests for user statistics."""

    def test_get_stats(self, app, db, sample_user):
        """Test getting user statistics."""
        with app.app_context():
            stats = UserService.get_stats()
            assert stats["total_users"] >= 1
            assert stats["active_users"] >= 1
