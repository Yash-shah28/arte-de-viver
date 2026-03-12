from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import get_db, engine
import models, schemas, crud

models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Art of Living Booking API",
    description="Create, read, update and delete bookings for AOL voice agent",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ──────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "running"}


# ── Create Booking ───────────────────────────────────────────────────────────
@app.post("/bookings/", response_model=schemas.BookingOut, status_code=201)
def create_booking(booking: schemas.BookingCreate, db: Session = Depends(get_db)):
    return crud.create_booking(db, booking)


# ── Get Booking by Email ─────────────────────────────────────────────────────
@app.get("/bookings/email/{email}", response_model=schemas.BookingOut)
def get_by_email(email: str, db: Session = Depends(get_db)):
    booking = crud.get_booking_by_email(db, email)
    if not booking:
        raise HTTPException(status_code=404, detail="No booking found for this email address")
    return booking


# ── Get Booking by ID ────────────────────────────────────────────────────────
@app.get("/bookings/{booking_id}", response_model=schemas.BookingOut)
def get_booking(booking_id: int, db: Session = Depends(get_db)):
    booking = crud.get_booking_by_id(db, booking_id)
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    return booking


# ── Update Booking (partial) ─────────────────────────────────────────────────
@app.patch("/bookings/email/{email}", response_model=schemas.BookingOut)
def update_booking(
    email: str,
    updates: schemas.BookingUpdate,
    db: Session = Depends(get_db),
):
    """
    Partially update a booking. Only the fields you include in the request
    body will be changed — omitted fields stay as they are.
    """
    booking = crud.update_booking(db, email, updates)
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    return booking


# ── Delete Booking ───────────────────────────────────────────────────────────
@app.delete("/bookings/email/{email}", status_code=204)
def delete_booking(email: str, db: Session = Depends(get_db)):
    """
    Permanently delete a booking by ID.
    Returns 204 No Content on success.
    """
    deleted = crud.delete_booking(db, email)
    if not deleted:
        raise HTTPException(status_code=404, detail="Booking not found")


# ── Get Instructor by Name ───────────────────────────────────────────────────
@app.get("/instructors/by-name/{name}", response_model=schemas.InstructorOut)
def get_instructor_by_name(name: str, db: Session = Depends(get_db)):
    instructor = crud.get_instructor_by_name(db, name)
    if not instructor:
        raise HTTPException(status_code=404, detail="Instructor not found")
    return instructor


# ── Get Instructor by Date ───────────────────────────────────────────────────
@app.get("/instructors/by-date/{class_date:path}", response_model=schemas.InstructorOut)
def get_instructor_by_date(class_date: str, db: Session = Depends(get_db)):
    """
    Returns the instructor assigned to a specific class date.
    Accepts:
      - YYYY-MM-DD    e.g. 2026-07-09
      - "D Month"     e.g. 9 July  (matches DB storage format)
      - "D de Mês"    e.g. 9 de julho  (Portuguese, auto-converted)
    Returns 404 if no instructor is scheduled for that date.
    """
    instructor = crud.get_instructor_by_date(db, class_date)
    if not instructor:
        raise HTTPException(status_code=404, detail="No instructor found for this date")
    return instructor