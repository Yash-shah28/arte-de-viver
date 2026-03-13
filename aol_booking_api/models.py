from sqlalchemy import Column, Integer, String, Date, DateTime
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.sql import func
from database import Base

class Booking(Base):
    __tablename__ = "bookings"

    id            = Column(Integer, primary_key=True, index=True)
    name          = Column(String, nullable=False)
    phone         = Column(String, nullable=False)
    email         = Column(String, nullable=False)
    birth_year    = Column(Integer, nullable=False)
    neighbourhood = Column(String, nullable=True)
    city          = Column(String, nullable=False)
    class_date    = Column(Date, nullable=False)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())

class Instructor(Base):
    __tablename__ = "instructors"

    name        = Column(String, primary_key=True, index=True)
    bio         = Column(String, nullable=True)
    event_dates = Column(ARRAY(String), nullable=True)  # stored as "9 July", "11 March", etc.