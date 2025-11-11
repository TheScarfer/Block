# Block.py
from fastapi import FastAPI, UploadFile, File as FastAPIFile, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import Column, Integer, String, Text, Float, create_engine, ForeignKey, Date
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt
import uvicorn
from openai import OpenAI
import os
import magic
import pdfplumber
from typing import Optional, List
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv("block.env")

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Database setup
Base = declarative_base()
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic Models
class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class GoalCreate(BaseModel):
    title: str

class GoalResponse(BaseModel):
    id: int
    title: str
    progress: float

    class Config:
        from_attributes = True

class ClassCreate(BaseModel):
    name: str
    link: Optional[str] = None
    syllabus: Optional[str] = None

class ClassResponse(BaseModel):
    id: int
    name: str
    link: Optional[str]
    syllabus: Optional[str]

    class Config:
        from_attributes = True

class EventCreate(BaseModel):
    title: str
    date: str
    description: Optional[str] = None

class EventResponse(BaseModel):
    id: int
    title: str
    date: str
    description: Optional[str]

    class Config:
        from_attributes = True

class FileResponse(BaseModel):
    id: int
    filename: str
    summary: str

    class Config:
        from_attributes = True

class FileListItem(BaseModel):
    id: int
    filename: str
    summary: str

    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    success: bool
    error: Optional[str] = None

# Database Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

    files = relationship("FileRecord", back_populates="owner")
    goals = relationship("Goal", back_populates="owner")
    classes = relationship("Class", back_populates="owner")
    events = relationship("CalendarEvent", back_populates="owner")

class FileRecord(Base):
    __tablename__ = "files"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    content = Column(Text)
    summary = Column(Text)
    owner_id = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="files")

class Goal(Base):
    __tablename__ = "goals"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    progress = Column(Float, default=0.0)
    owner_id = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="goals")

class Class(Base):
    __tablename__ = "classes"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    link = Column(String)
    syllabus = Column(Text)
    owner_id = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="classes")

class CalendarEvent(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    date = Column(Date)
    description = Column(Text)
    owner_id = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="events")

# Create tables
Base.metadata.create_all(bind=engine)

# Utility functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(db, username=username)
    if user is None:
        raise credentials_exception
    return user


# FastAPI App
app = FastAPI(
    title="AI Study Companion API", 
    version="1.0.0",
    max_request_size=120 * 1024 * 1024  # 120MB to be safe
)

# CORS - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
@app.get("/")
def root():
    return {"message": "AI Study Companion API is running", "status": "healthy"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/signup/", response_model=dict)
def signup(user_data: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    user = get_user(db, user_data.username)
    if user:
        raise HTTPException(
            status_code=400,
            detail="Username already exists"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    new_user = User(username=user_data.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {"message": "User created successfully", "username": user_data.username}

@app.post("/token/", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = get_user(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # System prompt for study assistance
        system_prompt = """You are an AI Study Assistant for the Study Companion app. You are helpful, encouraging, and educational. 
        You help students with:
        - Explaining difficult concepts in simple terms
        - Creating study plans and schedules
        - Answering academic questions
        - Generating quizzes and practice questions
        - Summarizing study materials
        - Providing study tips and techniques
        
        Keep responses clear, concise, and focused on learning. Be supportive and patient."""
        
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        # Add context if provided
        if request.context:
            messages.append({"role": "system", "content": f"Additional context: {request.context}"})
        
        # Add user message
        messages.append({"role": "user", "content": request.message})

        # Call OpenAI API
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        ai_text = completion.choices[0].message.content
        return ChatResponse(response=ai_text, success=True, error=None)
        
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return ChatResponse(
            response="I'm sorry, I'm having trouble processing your request right now. Please try again in a moment.",
            success=False,
            error=str(e)
        )

@app.post("/upload/", response_model=FileResponse)
def upload_file(
    file: UploadFile = FastAPIFile(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Check file size (10MB limit)
    MAX_FILE_SIZE = 10 * 1024 * 1024
    contents = file.file.read()
    
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File too large (max 10MB)"
        )
    
    # Detect file type
    try:
        file_type = magic.from_buffer(contents[:1024])
    except:
        file_type = "unknown"
    
    file.file.seek(0)

    # Extract text content
    text_content = ""
    if 'PDF' in file_type or file.filename.lower().endswith('.pdf'):
        try:
            with pdfplumber.open(file.file) as pdf:
                for page in pdf.pages:
                    text_content += (page.extract_text() or "") + "\n"
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error reading PDF: {str(e)}"
            )
    elif 'text' in file_type or file.filename.lower().endswith('.txt'):
        try:
            text_content = contents.decode("utf-8")
        except:
            text_content = contents.decode("latin-1")
    else:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Please upload PDF or text files."
        )

    # Generate AI summary
    try:
        # Limit text length for API
        text_for_summary = text_content[:3000]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful academic assistant. Provide a concise, study-focused summary highlighting key concepts, main ideas, and important details for effective learning."
                },
                {
                    "role": "user", 
                    "content": f"Please provide a comprehensive study summary of this content. Focus on key concepts, main ideas, and important details that would help a student understand and learn this material:\n\n{text_for_summary}"
                }
            ],
            max_tokens=500
        )
        summary = response.choices[0].message.content
    except Exception as e:
        print(f"Summary generation error: {str(e)}")
        summary = f"Summary generation unavailable. Content length: {len(text_content)} characters."

    # Save to database
    new_file = FileRecord(
        filename=file.filename,
        content=text_content[:5000],  # Store first 5000 chars
        summary=summary,
        owner_id=current_user.id
    )
    db.add(new_file)
    db.commit()
    db.refresh(new_file)

    return FileResponse(
        id=new_file.id,
        filename=new_file.filename,
        summary=new_file.summary
    )

@app.get("/files/", response_model=List[FileListItem])
def list_files(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    files = db.query(FileRecord).filter(FileRecord.owner_id == current_user.id).order_by(FileRecord.id.desc()).all()
    return [FileListItem(id=f.id, filename=f.filename, summary=f.summary) for f in files]

@app.post("/goals/", response_model=GoalResponse)
def create_goal(
    goal: GoalCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db_goal = Goal(title=goal.title, owner_id=current_user.id)
    db.add(db_goal)
    db.commit()
    db.refresh(db_goal)
    return GoalResponse(
        id=db_goal.id,
        title=db_goal.title,
        progress=db_goal.progress
    )

@app.get("/goals/", response_model=List[GoalResponse])
def get_goals(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    goals = db.query(Goal).filter(Goal.owner_id == current_user.id).all()
    return [GoalResponse(id=g.id, title=g.title, progress=g.progress) for g in goals]

@app.post("/classes/", response_model=ClassResponse)
def add_class(
    class_data: ClassCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    new_class = Class(
        name=class_data.name,
        link=class_data.link,
        syllabus=class_data.syllabus,
        owner_id=current_user.id
    )
    db.add(new_class)
    db.commit()
    db.refresh(new_class)
    return ClassResponse(
        id=new_class.id,
        name=new_class.name,
        link=new_class.link,
        syllabus=new_class.syllabus
    )

@app.get("/classes/", response_model=List[ClassResponse])
def get_classes(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    classes = db.query(Class).filter(Class.owner_id == current_user.id).all()
    return [ClassResponse(id=c.id, name=c.name, link=c.link, syllabus=c.syllabus) for c in classes]

@app.post("/events/", response_model=EventResponse)
def add_event(
    event_data: EventCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        event_date = datetime.strptime(event_data.date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use YYYY-MM-DD."
        )
    
    new_event = CalendarEvent(
        title=event_data.title,
        date=event_date,
        description=event_data.description,
        owner_id=current_user.id
    )
    db.add(new_event)
    db.commit()
    db.refresh(new_event)
    
    return EventResponse(
        id=new_event.id,
        title=new_event.title,
        date=new_event.date.isoformat(),
        description=new_event.description
    )

@app.get("/events/", response_model=List[EventResponse])
def get_events(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    events = db.query(CalendarEvent).filter(CalendarEvent.owner_id == current_user.id).all()
    return [EventResponse(id=e.id, title=e.title, date=e.date.isoformat(), description=e.description) for e in events]

# CORS preflight handlers
@app.options("/api/chat")
async def chat_options():
    return {"status": "ok"}

@app.options("/upload/")
async def upload_options():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting AI Study Companion API on http://localhost:{port}")
    print(f"📚 API Documentation: http://localhost:{port}/docs")
    uvicorn.run("Block:app", host="0.0.0.0", port=port, reload=True)
