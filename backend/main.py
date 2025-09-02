import sys, os
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.params import Depends
from typing import List,Optional
from backend.verdict_graph import run_verdict_graph_async

from backend.utils.websocketconn import ConnectionManager
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
import os
from auth.dependencies import get_current_user
load_dotenv()


from fastapi import FastAPI, UploadFile, status, BackgroundTasks, Form, HTTPException
from backend.database import schema
from backend.database.db import engine, Base, get_db
from backend.database import models
from backend.auth.auth import hash_password,verify_password,create_access_token

from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import uuid
import urllib.parse
ACCESS_TOKEN_EXPIRE_MINUTES=90

from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()  # initiate fastapi app--server

# ✅ Add CORS middleware if you need it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

@app.get('/')
async def health():
    return {'messages': "API is working well"}

@app.post('/create_user', response_model=schema.UserOut)
async def create_user(user: schema.UserCreate, db:Session= Depends(get_db)):
    hashed = hash_password(user.password)
    try:
        new_user= models.User(
            name=user.name,
            email=user.email,
            password_hash= hashed
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user
    except Exception as e:
        return f'error: {e}'


@app.post('/login', response_model=schema.Token)
async def login_user(user_in:schema.UserLogin, db:Session= Depends(get_db)):
    #check if user exists
    user= db.query(models.User).filter(models.User.email== user_in.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail= 'invalid email or password'

        )
    #generate_access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@app.get('/me', response_model=schema.UserOut)
def read_me(current_user: models.User= Depends(get_current_user)):
    return current_user

@app.post('/query')
def query_verdict(user_input:str,db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    answer= run_verdict_graph_async(user_input,db,current_user.id)
    return {"answer": answer}


@app.post('/test_query')
async def query_test(user_input: str, user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        return {"error": "User not found"}

    answer = await run_verdict_graph_async(user_input, db, user.id)
    return {"answer": answer}


@app.get('/message/{id}',response_model=List[schema.ConversationOut])
async def get_message_by_userid(id:int, db:Session=Depends(get_db)):
    message= db.query(models.Conversation).filter(models.Conversation.user_id== id).all()
    if not message:
        raise HTTPException(status_code=404, detail='Email not found')
    return message

manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str, db: Session = Depends(get_db)):
    from auth.dependencies import get_current_user
    # decode token and get user
    try:
        current_user = get_current_user(token=token, db=db)
        user_id = current_user.id
    except:
        await websocket.close(code=1008)
        return

    await manager.connect(websocket, user_id)

    try:
        while True:
            data = await websocket.receive_text()
            # call Verdict AI for response
            ai_response_text = await run_verdict_graph_async(data, db, user_id)

            # Create proper message object that matches your DB structure
            ai_message = {
                "message": ai_response_text,
                "role": "assistant",
                "created_at": datetime.now().isoformat(),
                "user_id": user_id
            }

            # Send JSON object back to frontend
            import json
            await manager.send_personal_message(json.dumps(ai_message), user_id)

    except WebSocketDisconnect:
        manager.disconnect(user_id)