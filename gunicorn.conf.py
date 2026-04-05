import threading
from app import _init_og, _ping

def post_fork(server, worker):
    threading.Thread(target=_init_og, daemon=True).start()
    threading.Thread(target=_ping, daemon=True).start()