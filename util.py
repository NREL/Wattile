import sys
import time
from datetime import datetime


start = time.time()
lasttime = time.time()


def prtime(*args, **kwargs):
    global lasttime
    print(" ".join(map(str, args)), '|time:', str(datetime.now()), '|', time.time() - start, 'secs from start',
          time.time() - lasttime, 'secs from last', **kwargs)
    lasttime = time.time()
    sys.stdout.flush()