from datetime import timedelta, datetime
from dotenv import load_dotenv
from jose import JWTError, jwt             # library for encoding/decoding JWT tokens
from passlib.context import CryptContext   # library for hashing + verifying passwords
import os
load_dotenv()
#jwt token config
SECRET_KEY = os.getenv('AUTH_SECRET_KEY')  # A secret string only you know (should come from env var in production)
ALGORITHM = "HS256"             # Algorithm used to encode/decode JWT
ACCESS_TOKEN_EXPIRE_MINUTES = 90

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
def hash_password(password: str) -> str:
    return pwd_context.hash(password)
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)