"""
session_manager.py - utility module for managing temporary in-memory user sorting sessions

This module provides session creation, session retrieval, automatic expiration handling,
session updates, and the cleanup of inactive sessions

Used by:
- bogosort.py
"""

from datetime import datetime, timedelta


class SessionManager:
    def __init__(self, timeout_minutes=30):
        self.sessions = {}
        self.timeout_minutes = timeout_minutes

    def get_session(self, session_id):
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if datetime.now() - session['last_access'] < timedelta(minutes=self.timeout_minutes):
                session['last_access'] = datetime.now()
                return session
            else:
                del self.sessions[session_id]
        return None

    def create_session(self, session_id):
        self.sessions[session_id] = {
            'state': None,
            'final_iteration': 0,
            'sorted': False,
            'error': None,
            'stop_flag': False,
            'algorithm': None,
            'last_access': datetime.now()
        }
        return self.sessions[session_id]

    def get_or_create_session(self, session_id):
        session = self.get_session(session_id)
        if session is None:
            session = self.create_session(session_id)
        return session

    def update_session(self, session_id, updates):
        session = self.get_or_create_session(session_id)
        session.update(updates)
        session['last_access'] = datetime.now()
        return session

    def cleanup_expired_sessions(self):
        current_time = datetime.now()
        expired = [
            sid for sid, session in self.sessions.items()
            if current_time - session['last_access'] > timedelta(minutes=self.timeout_minutes)
        ]
        for sid in expired:
            del self.sessions[sid]