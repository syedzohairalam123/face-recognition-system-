#!/usr/bin/env python3
"""
Seed Data Script
----------------
Populates the database with demo users for testing and demonstration.
Run once after setting up the database.

Usage:
    python seed_data.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.database import db
from app.models.user import User
from app.models.attendance import Attendance
from app.models.face_data import FaceData, CameraSource
from datetime import datetime, timedelta, date
import random


# Demo employee data
DEMO_USERS = [
    {
        "employee_id": "EMP001",
        "first_name": "Rahul",
        "last_name": "Sharma",
        "email": "rahul.sharma@company.com",
        "phone": "+91-9876543210",
        "department": "Engineering",
        "role": "employee",
    },
    {
        "employee_id": "EMP002",
        "first_name": "Priya",
        "last_name": "Patel",
        "email": "priya.patel@company.com",
        "phone": "+91-9876543211",
        "department": "Engineering",
        "role": "employee",
    },
    {
        "employee_id": "EMP003",
        "first_name": "Amit",
        "last_name": "Kumar",
        "email": "amit.kumar@company.com",
        "phone": "+91-9876543212",
        "department": "Marketing",
        "role": "employee",
    },
    {
        "employee_id": "EMP004",
        "first_name": "Sneha",
        "last_name": "Reddy",
        "email": "sneha.reddy@company.com",
        "phone": "+91-9876543213",
        "department": "HR",
        "role": "manager",
    },
    {
        "employee_id": "EMP005",
        "first_name": "Vikram",
        "last_name": "Singh",
        "email": "vikram.singh@company.com",
        "phone": "+91-9876543214",
        "department": "Finance",
        "role": "employee",
    },
    {
        "employee_id": "EMP006",
        "first_name": "Anjali",
        "last_name": "Gupta",
        "email": "anjali.gupta@company.com",
        "phone": "+91-9876543215",
        "department": "Engineering",
        "role": "admin",
    },
    {
        "employee_id": "EMP007",
        "first_name": "Ravi",
        "last_name": "Verma",
        "email": "ravi.verma@company.com",
        "phone": "+91-9876543216",
        "department": "Sales",
        "role": "employee",
    },
    {
        "employee_id": "EMP008",
        "first_name": "Neha",
        "last_name": "Joshi",
        "email": "neha.joshi@company.com",
        "phone": "+91-9876543217",
        "department": "Design",
        "role": "employee",
    },
]


def seed_database():
    """Populate database with demo data."""
    app = create_app("development")

    with app.app_context():
        # Check if users already exist
        existing = User.query.count()
        if existing > 0:
            print(f"[INFO] Database already has {existing} users. Skipping seed.")
            return

        print("[SEED] Seeding database with demo users...")

        # Create camera sources
        cameras = [
            CameraSource(name="Main Entrance", location="Building A - Lobby", camera_index=0),
            CameraSource(name="Office Door", location="Building B - Floor 2", camera_index=1),
        ]
        for cam in cameras:
            db.session.add(cam)
        db.session.commit()
        print(f"[OK] Created {len(cameras)} camera sources")

        # Create users
        users = []
        for user_data in DEMO_USERS:
            user = User(
                employee_id=user_data["employee_id"],
                first_name=user_data["first_name"],
                last_name=user_data["last_name"],
                email=user_data["email"],
                phone=user_data.get("phone"),
                department=user_data["department"],
                role=user_data["role"],
                is_active=True,
                face_registered=False,
            )
            # Set default password
            user.set_password("12345678")
            db.session.add(user)
            users.append(user)

        db.session.commit()
        print(f"[OK] Created {len(users)} demo users with default password: 12345678")

        # Create some sample attendance records for today
        today = date.today()
        now = datetime.utcnow()
        attendance_count = 0

        for user in random.sample(users, min(5, len(users))):
            check_in_hour = random.randint(8, 10)
            check_in_minute = random.randint(0, 59)
            check_in_time = now.replace(hour=check_in_hour, minute=check_in_minute, second=0, microsecond=0)

            record = Attendance(
                user_id=user.id,
                attendance_date=today,
                check_in_time=check_in_time,
                status="present",
                confidence_score=random.uniform(0.85, 0.99),
                camera_source="Main Entrance",
            )

            # Some users have checked out
            if random.random() > 0.4:
                check_out_hour = check_in_hour + random.randint(4, 8)
                if check_out_hour <= 17:
                    record.check_out_time = check_in_time.replace(hour=check_out_hour)
                    record.status = "checked_out"

            db.session.add(record)
            attendance_count += 1

        db.session.commit()
        print(f"[OK] Created {attendance_count} sample attendance records for today")

        # Create some historical records (last 7 days)
        historical_count = 0
        for days_ago in range(1, 8):
            day = today - timedelta(days=days_ago)
            if day.weekday() >= 5:  # Skip weekends
                continue

            for user in random.sample(users, random.randint(3, 6)):
                check_in_hour = random.randint(8, 10)
                check_in_time = datetime.combine(day, datetime.min.time().replace(hour=check_in_hour, minute=random.randint(0, 59)))

                record = Attendance(
                    user_id=user.id,
                    attendance_date=day,
                    check_in_time=check_in_time,
                    status="present",
                    confidence_score=random.uniform(0.80, 0.99),
                    camera_source=random.choice(["Main Entrance", "Office Door"]),
                )

                check_out_hour = check_in_hour + random.randint(4, 8)
                if check_out_hour <= 17:
                    record.check_out_time = check_in_time.replace(hour=check_out_hour)
                    record.status = "checked_out"

                db.session.add(record)
                historical_count += 1

        db.session.commit()
        print(f"[OK] Created {historical_count} historical attendance records")

        print("\n[DONE] Database seeded successfully!")
        print("\n[DEMO] Demo Users:")
        print("-" * 65)
        print(f"{'ID':<10} {'Name':<25} {'Department':<15} {'Role':<10}")
        print("-" * 65)
        for user in users:
            print(f"{user.employee_id:<10} {user.full_name:<25} {user.department:<15} {user.role:<10}")
        print("-" * 65)
        print("\n[PASSWORD] Default password for all users: 12345678")
        print("[RUN] python run.py to start the application")


if __name__ == "__main__":
    seed_database()
