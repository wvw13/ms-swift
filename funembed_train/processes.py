"""Scoped Linux subprocess-group ownership and interrupt cleanup."""
import os,signal,time,threading
from contextlib import contextmanager

class Interrupted(KeyboardInterrupt):
 def __init__(self,signum):self.signum=signum;super().__init__(signal.Signals(signum).name)

@contextmanager
def interrupt_handlers():
 if threading.current_thread() is not threading.main_thread():
  yield;return
 old={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
 def handle(s,frame):raise Interrupted(s)
 try:
  for s in old:signal.signal(s,handle)
  yield
 finally:
  for s,h in old.items():signal.signal(s,h)

def stop_group(p,grace=30):
 # Only the group created with start_new_session=True by this invocation.
 old={}
 if threading.current_thread() is threading.main_thread():
  old={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
  for s in old:signal.signal(s,signal.SIG_IGN)
 def send(sig):
  try:os.killpg(p.pid,sig)
  except ProcessLookupError:pass
 def alive():
  try:os.killpg(p.pid,0);return True
  except ProcessLookupError:return False
 try:
  send(signal.SIGTERM);end=time.monotonic()+grace
  while alive() and time.monotonic()<end:p.poll();time.sleep(.05)
  if alive():send(signal.SIGKILL)
  p.wait()
 finally:
  for s,h in old.items():signal.signal(s,h)
