from pydantic import BaseModel
from typing import Optional, List
from datetime import date, datetime


class BookingCreate(BaseModel):
    name          : str
    phone         : str
    email         : str
    birth_year    : int           # 4-digit year e.g. 1995
    neighbourhood : Optional[str] = None
    city          : str
    class_date    : date          # Format: YYYY-MM-DD  e.g. 2026-03-15


class BookingUpdate(BaseModel):
    """All fields optional — only provided fields are updated (PATCH behaviour)."""
    name          : Optional[str]  = None
    phone         : Optional[str]  = None
    email         : Optional[str]  = None
    birth_year    : Optional[int]  = None
    neighbourhood : Optional[str]  = None
    city          : Optional[str]  = None
    class_date    : Optional[date] = None


class BookingOut(BookingCreate):
    id         : int
    created_at : datetime

    class Config:
        from_attributes = True


class InstructorOut(BaseModel):
    name        : str
    bio         : Optional[str] = None
    # event_dates : Optional[List[date]] = None

    class Config:
        from_attributes = True