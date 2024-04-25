
import importlib
import logging
import os
import sys
import logging
from pathlib import Path
from typing import Any, Union

from pydantic import BaseModel

logger = logging.getLogger(__name__)

class PydevConfig(BaseModel):
    sys_path: str
    client: str
    port: int
    client_access_token: str
    ppid: int


def get_debugpy() -> Any:
    try:
        return importlib.import_module("debugpy")
    except Exception:
        return None


def break_in_vscode() -> None:
    """Breaks into the debugger if it is connected."""
    debugpy = get_debugpy()
    if debugpy is None:
        return
    if debugpy.is_client_connected():
        debugpy.breakpoint()


def connect_to_pydev(config: Union[PydevConfig, None]) -> None:
    print(config)
    if config is None:
        return

    sys_path = config.sys_path
    sys.path.append(sys_path)

    logging.info("Added debupy sys path %s", sys_path)

    debugpy = get_debugpy()
    if debugpy is None:
        logger.exception("debugpy was not available to the raylet even after adding the required syspath. Bailing out.")
        return

    logging.info("Imported debugpy")

    if debugpy.is_client_connected():
        logger.warning("debugpy client already connected.")
        return

    client = config.client
    port = config.port
    client_access_token = config.client_access_token

    pydevd = debugpy.server.api.pydevd
    pydevd.SetupHolder.setup = {
        "ppid": config.ppid,
        "client": client,
        "port": port,
        "client-access-token": client_access_token,
    }

    logging.info("Connecting to debugpy at %s: %s", client, port)
    debugpy.connect([client, port], access_token=client_access_token)

    logger.info("Waiting for debugger connection.")
    debugpy.wait_for_client()
    logger.info("Connected to the debugger.")


def create_pydev_config() -> Union[PydevConfig, None]:
    debugpy = get_debugpy()
    if debugpy is None:
        logger.info("Pydev is not available, vs code debug support is not available.")
        return None

    if not debugpy.is_client_connected():
        logger.info("No VS code was available during boot.")
        return None

    pydevd = debugpy.server.api.pydevd
    setup = pydevd.SetupHolder.setup

    pydev_config = PydevConfig(
        ppid=os.getpid(),
        sys_path=str(Path(debugpy.__file__).resolve().parents[1]),
        client=setup.get("client"),
        port=setup.get("port"),
        client_access_token=setup.get("client-access-token"),
    )

    logger.info("Created VSCode pydev config %s", pydev_config)
    return pydev_config
