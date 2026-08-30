"""
User Service
------------
Handles all user/employee CRUD operations and business logic.
"""

import logging
import os
from typing import Optional, List, Dict, Tuple

from app.database import db
from app.models.user import User

logger = logging.getLogger(__name__)


class UserService:
    """Service layer for user management operations."""

    @staticmethod
    def create_user(data: Dict) -> Tuple[Optional[User], str]:
        """
        Create a new user/employee.

        Args:
            data: Dictionary with user fields (employee_id, first_name, etc.)

        Returns:
            Tuple of (User object or None, status message)
        """
        try:
            # Check for duplicate employee_id
            if User.query.filter_by(employee_id=data["employee_id"]).first():
                return None, "Employee ID already exists"

            # Check for duplicate email
            if User.query.filter_by(email=data["email"]).first():
                return None, "Email already registered"

            user = User(
                employee_id=data["employee_id"].strip(),
                first_name=data["first_name"].strip(),
                last_name=data["last_name"].strip(),
                email=data["email"].strip().lower(),
                phone=data.get("phone", "").strip() if data.get("phone") else None,
                department=data.get("department", "").strip() if data.get("department") else None,
                role=data.get("role", "employee"),
            )

            db.session.add(user)
            db.session.commit()

            logger.info(f"User created: {user.employee_id} ({user.full_name})")
            return user, "User created successfully"

        except KeyError as e:
            db.session.rollback()
            logger.error(f"Missing required field: {e}")
            return None, f"Missing required field: {e}"
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating user: {e}")
            return None, f"Error creating user: {str(e)}"

    @staticmethod
    def get_user_by_id(user_id: int) -> Optional[User]:
        """Get a user by their database ID."""
        return User.query.get(user_id)

    @staticmethod
    def get_user_by_employee_id(employee_id: str) -> Optional[User]:
        """Get a user by their employee ID."""
        return User.query.filter_by(employee_id=employee_id.strip()).first()

    @staticmethod
    def get_all_users(active_only: bool = True) -> List[User]:
        """Get all users, optionally filtering to active only."""
        query = User.query
        if active_only:
            query = query.filter_by(is_active=True)
        return query.order_by(User.created_at.desc()).all()

    @staticmethod
    def update_user(user_id: int, data: Dict) -> Tuple[Optional[User], str]:
        """
        Update user information.

        Args:
            user_id: The user's database ID
            data: Dictionary of fields to update

        Returns:
            Tuple of (User object or None, status message)
        """
        try:
            user = User.query.get(user_id)
            if not user:
                return None, "User not found"

            # Check email uniqueness if changing
            if "email" in data and data["email"] != user.email:
                if User.query.filter_by(email=data["email"].strip().lower()).first():
                    return None, "Email already registered"

            if "first_name" in data:
                user.first_name = data["first_name"].strip()
            if "last_name" in data:
                user.last_name = data["last_name"].strip()
            if "email" in data:
                user.email = data["email"].strip().lower()
            if "phone" in data:
                user.phone = data["phone"].strip() if data["phone"] else None
            if "department" in data:
                user.department = data["department"].strip() if data["department"] else None
            if "role" in data:
                user.role = data["role"]
            if "is_active" in data:
                user.is_active = data["is_active"]

            db.session.commit()
            logger.info(f"User updated: {user.employee_id}")
            return user, "User updated successfully"

        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating user {user_id}: {e}")
            return None, f"Error updating user: {str(e)}"

    @staticmethod
    def delete_user(user_id: int) -> str:
        """
        Soft-delete a user (set inactive).

        Returns:
            Status message
        """
        try:
            user = User.query.get(user_id)
            if not user:
                return "User not found"

            user.is_active = False
            db.session.commit()
            logger.info(f"User deactivated: {user.employee_id}")
            return f"User {user.full_name} deactivated successfully"

        except Exception as e:
            db.session.rollback()
            logger.error(f"Error deactivating user {user_id}: {e}")
            return f"Error deactivating user: {str(e)}"

    @staticmethod
    def activate_user(user_id: int) -> Tuple[Optional[User], str]:
        """
        Reactivate a previously deactivated user.

        Returns:
            Tuple of (User object or None, status message)
        """
        try:
            user = User.query.get(user_id)
            if not user:
                return None, "User not found"

            user.is_active = True
            db.session.commit()
            logger.info(f"User reactivated: {user.employee_id}")
            return user, f"User {user.full_name} reactivated successfully"

        except Exception as e:
            db.session.rollback()
            logger.error(f"Error reactivating user {user_id}: {e}")
            return None, f"Error reactivating user: {str(e)}"

    @staticmethod
    def hard_delete_user(user_id: int) -> Tuple[bool, str]:
        """
        Permanently delete a user, their attendance history, and face data.

        Returns:
            Tuple of (success boolean, status message)
        """
        try:
            user = User.query.get(user_id)
            if not user:
                return False, "User not found"

            employee_label = f"{user.full_name} ({user.employee_id})"

            # Remove face encoding file from disk if present
            face_path = user.face_data_path
            db.session.delete(user)  # Attendance records cascade via relationship
            db.session.commit()

            if face_path:
                try:
                    if os.path.exists(face_path):
                        os.remove(face_path)
                except OSError as e:
                    logger.warning(f"Could not remove face file for user {user_id}: {e}")

            logger.info(f"User permanently deleted: {employee_label}")
            return True, f"User {employee_label} permanently deleted"

        except Exception as e:
            db.session.rollback()
            logger.error(f"Error deleting user {user_id}: {e}")
            return False, f"Error deleting user: {str(e)}"

    @staticmethod
    def mark_face_registered(user_id: int, face_data_path: str) -> Tuple[bool, str]:
        """
        Mark a user's face as registered and store the face data path.

        Args:
            user_id: The user's database ID
            face_data_path: Path to the stored face encoding file

        Returns:
            Tuple of (success boolean, message)
        """
        try:
            user = User.query.get(user_id)
            if not user:
                return False, "User not found"

            user.face_registered = True
            user.face_data_path = face_data_path
            db.session.commit()

            logger.info(f"Face registered for user: {user.employee_id}")
            return True, "Face registration completed"

        except Exception as e:
            db.session.rollback()
            logger.error(f"Error marking face registered for user {user_id}: {e}")
            return False, f"Error: {str(e)}"

    @staticmethod
    def get_users_needing_registration() -> List[User]:
        """Get all active users who haven't registered their face yet."""
        return User.query.filter_by(is_active=True, face_registered=False).all()

    @staticmethod
    def get_registered_users() -> List[User]:
        """Get all active users with registered faces."""
        return User.query.filter_by(is_active=True, face_registered=True).all()

    @staticmethod
    def search_users(query: str, include_inactive: bool = True) -> List[User]:
        """
        Search users by name, employee_id, or email.
        By default includes inactive users so admins can find and reactivate them.
        """
        search_term = f"%{query}%"
        q = User.query.filter(
            db.or_(
                User.first_name.ilike(search_term),
                User.last_name.ilike(search_term),
                User.employee_id.ilike(search_term),
                User.email.ilike(search_term),
            )
        )
        if not include_inactive:
            q = q.filter_by(is_active=True)
        return q.order_by(User.created_at.desc()).all()

    @staticmethod
    def get_stats() -> Dict:
        """Get user statistics."""
        total = User.query.count()
        active = User.query.filter_by(is_active=True).count()
        registered = User.query.filter_by(is_active=True, face_registered=True).count()
        unregistered = User.query.filter_by(is_active=True, face_registered=False).count()

        return {
            "total_users": total,
            "active_users": active,
            "face_registered": registered,
            "face_pending": unregistered,
        }
