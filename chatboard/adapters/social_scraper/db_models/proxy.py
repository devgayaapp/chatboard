from enum import Enum
from typing import Optional
from pydantic import BaseModel


class ProxyType(str, Enum):
    HTTP = "http"
    SOCKS5 = "socks5"


class ProxyService(str, Enum):
    PROXYEMPIRE = "proxyempire"
    CHEAPPROXY = "cheapproxy"


class Proxy(BaseModel):
    ip: str = None
    port: int = None
    username: str = None
    password: str = None
    iso: Optional[str] = None
    sticky_session_key: Optional[str] = None
    proxy_type: Optional[ProxyType] = None
    proxy_service: Optional[ProxyService] = None
