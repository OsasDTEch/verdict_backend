from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime

# ---------------------------
# User Schemas
# ---------------------------
class UserBase(BaseModel):
    name: str = Field(..., example="John Doe")
    email: EmailStr = Field(..., example="john@example.com")
class UserLogin(BaseModel):
    email: EmailStr = Field(..., example="john@example.com")
    password: str = Field(..., min_length=6, example="strongpassword123")

class UserCreate(UserBase):
    password: str = Field(..., min_length=6, example="strongpassword123")

class UserOut(BaseModel):
    id: int
    name: str
    email: str
    created_at: datetime

    model_config = {"from_attributes": True}  # replaces orm_mode


class Token(BaseModel):
    access_token: str
    token_type: str

# ---------------------------
# Conversation Schemas
# ---------------------------
class ConversationBase(BaseModel):
    role: str = Field(..., example="user")   # user | assistant
    message: str = Field(..., example="Hello, how are you?")

class ConversationCreate(ConversationBase):
    pass

class ConversationOut(ConversationBase):
    id: int
    user_id: int
    timestamp: datetime

    class Config:
        from_attributes = True


