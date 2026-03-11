from sqlalchemy.orm import Session
from datetime import date
import models, schemas


def create_booking(db: Session, booking: schemas.BookingCreate):
    db_booking = models.Booking(**booking.dict())
    db.add(db_booking)
    db.commit()
    db.refresh(db_booking)
    return db_booking


def get_booking_by_id(db: Session, booking_id: int):
    return db.query(models.Booking).filter(models.Booking.id == booking_id).first()


def get_booking_by_email(db: Session, email: str):
    return db.query(models.Booking).filter(models.Booking.email == email).first()


def get_instructor_by_name(db: Session, name: str):
    return db.query(models.Instructor).filter(
        models.Instructor.name.ilike(f"%{name}%")
    ).first()


def get_instructor_by_date(db: Session, class_date: date):
    """Return the instructor whose event_dates array contains the given date."""
    return db.query(models.Instructor).filter(
        models.Instructor.event_dates.any(class_date)
    ).first()


def update_booking(db: Session, email: str, updates: schemas.BookingUpdate):
    db_booking = get_booking_by_email(db, email)
    if not db_booking:
        return None
    # Only update fields that were explicitly provided (not None)
    for field, value in updates.dict(exclude_unset=True).items():
        setattr(db_booking, field, value)
    db.commit()
    db.refresh(db_booking)
    return db_booking


def delete_booking(db: Session, email: str):
    db_booking = get_booking_by_email(db, email)
    if not db_booking:
        return False
    db.delete(db_booking)
    db.commit()
    return True