import toml
import os

conf_file_env = 'AIMETRICS_CONF'
conf_file_default = "conf.toml"

def get_conf(conf_file=None):
    """
    Convenience method to get aimetrics configuration

    Arguments
    ---------
    conf_file : str
        Path to the TOML configuration file [default: None]

        The config file is resolved in the following order:

        1. If conf_file is a file path, that file is used.
        2. If the environment variable 'AIMETRICS_CONF' is set and points to an
           existing toml file, this file is used for configuration.
        3. If the file "conf.toml" exists in the pwd, it is used.
        4. The empty dictionary is returned.

    """

    if conf_file is None:
        conf_file = os.environ.get(conf_file_env, None)
        if conf_file is None and os.path.exists(conf_file_default):
            conf_file = conf_file_default
        else:
            return {}
    with open(conf_file) as f:
        return toml.loads(f.read())
