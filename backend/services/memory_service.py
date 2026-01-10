import sqlite3
import threading
import os
import time
from pathlib import Path
from watchfiles import watch, Change
from config import BASE_DIR

class MemoryService:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = BASE_DIR / "memory.db"
        self.db_path = str(db_path)
        self.lock = threading.Lock()
        self._init_db()
        self.watching = False
        
    def _init_db(self):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE,
                    filename TEXT,
                    extension TEXT,
                    size INTEGER,
                    modified REAL
                )
            ''')
            # Index for fast searching
            c.execute('CREATE INDEX IF NOT EXISTS idx_filename ON files(filename)')
            conn.commit()
            conn.close()

    def _get_file_info(self, path):
        try:
            p = Path(path)
            if not p.exists() or not p.is_file():
                return None
            stat = p.stat()
            return {
                'path': str(p.resolve()),
                'filename': p.name,
                'extension': p.suffix.lower(),
                'size': stat.st_size,
                'modified': stat.st_mtime
            }
        except Exception:
            return None

    def upsert_file(self, path):
        info = self._get_file_info(path)
        if not info:
            self.remove_file(path) # If it doesn't exist/isn't file, remove it
            return

        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''
                INSERT INTO files (path, filename, extension, size, modified)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    filename=excluded.filename,
                    extension=excluded.extension,
                    size=excluded.size,
                    modified=excluded.modified
            ''', (info['path'], info['filename'], info['extension'], info['size'], info['modified']))
            conn.commit()
            conn.close()

    def remove_file(self, path):
        # Resolve path to ensure consistency if possible, though deleted file can't be resolved easily
        # We rely on the string path passed from watcher
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('DELETE FROM files WHERE path = ?', (str(Path(path).resolve()),))
            conn.commit()
            conn.close()

    def scan_directory(self, root_path):
        print(f"[Memory] Scanning {root_path}...")
        count = 0
        batch = []
        BATCH_SIZE = 100
        
        root = Path(root_path)
        if not root.exists():
            print(f"[Memory] Path {root_path} does not exist.")
            return

        # Clear existing entries for this root? 
        # For now, we mix everything. A full re-scan might want to clean up old entries from this dir.
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        for p in root.rglob('*'):
            if p.is_file():
                info = self._get_file_info(p)
                if info:
                    batch.append((info['path'], info['filename'], info['extension'], info['size'], info['modified']))
                    count += 1
                    
                    if len(batch) >= BATCH_SIZE:
                        c.executemany('''
                            INSERT INTO files (path, filename, extension, size, modified)
                            VALUES (?, ?, ?, ?, ?)
                            ON CONFLICT(path) DO UPDATE SET
                                filename=excluded.filename,
                                extension=excluded.extension,
                                size=excluded.size,
                                modified=excluded.modified
                        ''', batch)
                        batch = []
        
        if batch:
            c.executemany('''
                INSERT INTO files (path, filename, extension, size, modified)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    filename=excluded.filename,
                    extension=excluded.extension,
                    size=excluded.size,
                    modified=excluded.modified
            ''', batch)
            
        conn.commit()
        conn.close()
        print(f"[Memory] Scanned {count} files in {root_path}.")

    def start_watcher(self, root_path):
        if self.watching:
            return
        
        self.watching = True
        
        def run_watch():
            print(f"[Memory] Starting watcher on {root_path}")
            # Watch recursively
            try:
                for changes in watch(root_path):
                    for change, path in changes:
                        try:
                            if change == Change.added or change == Change.modified:
                                self.upsert_file(path)
                            elif change == Change.deleted:
                                self.remove_file(path)
                        except Exception as e:
                            print(f"[Memory] Error processing change {path}: {e}")
            except Exception as e:
                print(f"[Memory] Watcher crashed: {e}")
                self.watching = False

        t = threading.Thread(target=run_watch, daemon=True)
        t.start()

    def search(self, query, limit=20):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            # Simple fuzzy search
            like_query = f"%{query}%"
            c.execute('''
                SELECT path, size, modified FROM files 
                WHERE filename LIKE ? OR path LIKE ?
                ORDER BY modified DESC
                LIMIT ?
            ''', (like_query, like_query, limit))
            results = c.fetchall()
            conn.close()
            
            return [{'path': r[0], 'size': r[1], 'modified': r[2]} for r in results]

memory_service = MemoryService()
