import sqlite3
import os

DB_PATH = "data/calls.db"

def view_data():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print("\n--- RECENT CALL SESSIONS ---")
    sessions = cursor.execute("SELECT * FROM call_sessions ORDER BY created_at DESC LIMIT 5").fetchall()
    
    if not sessions:
        print("No sessions found.")
    
    for sess in sessions:
        print(f"\nID: {sess['session_id']}")
        print(f"Caller: {sess['caller_number']} | Duration: {sess['call_duration_s']:.1f}s")
        print(f"Sentiment: {sess['sentiment']} | Summary: {sess['summary']}")
        
        print("Transcript:")
        turns = cursor.execute("SELECT speaker, message FROM conversations WHERE session_id = ? ORDER BY turn_index", (sess['session_id'],)).fetchall()
        for turn in turns:
            print(f"  [{turn['speaker'].upper()}] {turn['message']}")

    conn.close()

if __name__ == "__main__":
    view_data()
