import sys
import os
import tqdm
import tqdm.auto
import tqdm.std
import threading

# Global callback reference
_progress_callback = None
_active_bars = set()
_bars_lock = threading.Lock()

def set_progress_callback(callback):
    global _progress_callback
    _progress_callback = callback

class SilentFile:
    def write(self, x): pass
    def flush(self): pass

class AggregatedMonitor:
    @staticmethod
    def report():
        if not _progress_callback:
            return

        try:
            with _bars_lock:
                current_bars = list(_active_bars)
            
            total_bytes = 0
            downloaded_bytes = 0
            valid_bars = 0
            rate_sum = 0
            
            for bar in current_bars:
                if hasattr(bar, 'disable') and bar.disable:
                    continue
                    
                unit = getattr(bar, 'unit', '')
                b_total = bar.total if bar.total else 0
                b_n = bar.n
                
                # Filter out small files/counters
                if unit and 'it' in unit and b_total < 1000:
                    continue
                    
                total_bytes += b_total
                downloaded_bytes += b_n
                
                info = bar.format_dict if hasattr(bar, 'format_dict') else {}
                rate_sum += info.get('rate', 0) if info.get('rate') else 0
                
                valid_bars += 1

            if valid_bars > 0:
                _progress_callback(downloaded_bytes, total_bytes, rate_sum)
                
        except Exception as e:
            pass

class WrapperTqdm(tqdm.auto.tqdm):
    def __init__(self, *args, **kwargs):
        # FORCE ENABLE: Even if no TTY, we want callbacks
        kwargs['disable'] = False
        
        # SQUELCH OUTPUT: Reduce log spam by writing to dummy if not specified
        if 'file' not in kwargs:
            kwargs['file'] = SilentFile()
            
        super().__init__(*args, **kwargs)
        with _bars_lock:
            _active_bars.add(self)
            
    def close(self):
        with _bars_lock:
            if self in _active_bars:
                _active_bars.remove(self)
        super().close()
        AggregatedMonitor.report()

    def update(self, n=1):
        super().update(n)
        AggregatedMonitor.report()

# Apply Patch
def apply_patch():
    if tqdm.tqdm != WrapperTqdm:
        tqdm.tqdm = WrapperTqdm
        tqdm.auto.tqdm = WrapperTqdm
        tqdm.std.tqdm = WrapperTqdm
        
        if 'tqdm' in sys.modules: sys.modules['tqdm'].tqdm = WrapperTqdm
        if 'tqdm.auto' in sys.modules: sys.modules['tqdm.auto'].tqdm = WrapperTqdm
        if 'tqdm.std' in sys.modules: sys.modules['tqdm.std'].tqdm = WrapperTqdm

apply_patch()
