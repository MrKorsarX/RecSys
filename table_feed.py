from sqlalchemy import Column, Integer, String, TIMESTAMP, ForeignKey
from sqlalchemy.orm import relationship

from database import Base, SessionLocal
from table_user import User
from table_post import Post


class Feed(Base):
    __tablename__ = "feed_action"
    user_id = Column(Integer, ForeignKey(User.id), primary_key=True,)
    post_id = Column(Integer, ForeignKey(Post.id))
    action = Column(String)
    time = Column(TIMESTAMP)

    user = relationship(User)
    post = relationship(Post)
