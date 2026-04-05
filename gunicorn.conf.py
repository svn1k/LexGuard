import threading
from app import _init_og, _ping

# Запускаем при загрузке конфига — до форка воркеров
threading.Thread(target=_init_og, daemon=True).start()
threading.Thread(target=_ping, daemon=True).start()

worker_class = "gevent"
workers = 1
timeout = 300