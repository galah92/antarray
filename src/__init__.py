from jax.experimental.compilation_cache import compilation_cache as cc

# Persistent Jax compilation cache: https://docs.jax.dev/en/latest/persistent_compilation_cache.html
cc.set_cache_dir("/tmp/jax_cache")
