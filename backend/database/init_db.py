from backend.database.session import engine
from backend.database.base import Base
from backend.database import models

def init_db():
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")

if __name__ == "__main__":
    init_db()
