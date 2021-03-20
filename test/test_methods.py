import os
import jax
from jax.experimental.stax import serial, Dense
import helx


def pmodule_test():
    jax.config.update("jax_platform_name", "cpu")
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    n_devices = jax.local_device_count()
    print(n_devices)

    @helx.methods.pmodule
    def m(a, b):
        return serial(Dense(a), Dense(b))

    m(2, 3)


if __name__ == "__main__":
    pmodule_test()