from sqlalchemy.orm import Session
from sqlalchemy import cast, String
from sqlalchemy.dialects.postgresql import ARRAY
from datetime import date, datetime
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


def _to_day_month_label(class_date: str) -> str:
    """
    Convert any reasonable date input to the "D Month" format stored in event_dates.
    Accepts:
      - "YYYY-MM-DD"  → "9 July"
      - "9 July"      → "9 July"  (pass-through)
      - "July 9"      → "9 July"
      - "9 de julho"  → "9 July"
    """
    PT_MONTHS = {
        "janeiro": "January", "fevereiro": "February", "março": "March",
        "marco": "March", "abril": "April", "maio": "May", "junho": "June",
        "julho": "July", "agosto": "August", "setembro": "September",
        "outubro": "October", "novembro": "November", "dezembro": "December",
    }

    s = class_date.strip()

    # 1. ISO format YYYY-MM-DD
    try:
        d = datetime.strptime(s, "%Y-%m-%d")
        return f"{d.day} {d.strftime('%B')}"   # e.g. "9 July"
    except ValueError:
        pass

    # 2. Translate Portuguese month names to English
    s_lower = s.lower()
    for pt, en in PT_MONTHS.items():
        if pt in s_lower:
            s_lower = s_lower.replace(pt, en.lower())
            s = s_lower
            break

    # Remove filler words
    for filler in (" de ", " do ", " da "):
        s = s.replace(filler, " ")
    s = s.strip()

    # 3. "D Month" or "D MonthName" — already correct format
    try:
        d = datetime.strptime(s, "%d %B")
        return f"{d.day} {d.strftime('%B')}"
    except ValueError:
        pass
    try:
        d = datetime.strptime(s, "%d %b")
        return f"{d.day} {d.strftime('%B')}"
    except ValueError:
        pass

    # 4. "Month D" or "MonthName D"
    try:
        d = datetime.strptime(s, "%B %d")
        return f"{d.day} {d.strftime('%B')}"
    except ValueError:
        pass
    try:
        d = datetime.strptime(s, "%b %d")
        return f"{d.day} {d.strftime('%B')}"
    except ValueError:
        pass

    # 5. Return as-is and let the query simply not match
    return s


def get_instructor_by_date(db: Session, class_date: str):
    """
    Return the instructor whose event_dates array contains the given date.
    class_date can be YYYY-MM-DD (from agent) or "9 July" (already normalised).
    Matching is case-insensitive against the "D Month" strings stored in the DB.
    """
    label = _to_day_month_label(class_date)   # e.g. "9 July"

    # Use PostgreSQL ANY with case-insensitive comparison
    # event_dates is character varying[] so we unnest and ilike
    from sqlalchemy import func, text

    return (
        db.query(models.Instructor)
        .filter(
            # Check if any element in the array matches the label (case-insensitive)
            models.Instructor.event_dates.any(
                # cast each element to text and compare
                label
            )
        )
        .first()
    ) or (
        # Fallback: manual unnest approach for case-insensitive match
        db.query(models.Instructor)
        .filter(
            func.lower(label).op("= ANY")(
                func.array(
                    func.unnest(models.Instructor.event_dates)
                )
            )
        )
        .first()
    ) or _get_instructor_by_date_fallback(db, label)


def _get_instructor_by_date_fallback(db: Session, label: str):
    """
    Pure-Python fallback: load all instructors and match in Python.
    Used when SQL ANY operators fail due to type casting edge cases.
    """
    label_lower = label.strip().lower()
    instructors = db.query(models.Instructor).filter(
        models.Instructor.event_dates.isnot(None)
    ).all()
    for instructor in instructors:
        for event_date in (instructor.event_dates or []):
            if event_date and event_date.strip().lower() == label_lower:
                return instructor
    return None


def update_booking(db: Session, email: str, updates: schemas.BookingUpdate):
    db_booking = get_booking_by_email(db, email)
    if not db_booking:
        return None
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