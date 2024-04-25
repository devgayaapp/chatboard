


from components.etl.util.debug_config import create_pydev_config, connect_to_pydev



pydev_config = create_pydev_config()



def connect_to_debug():
    connect_to_pydev(pydev_config)
