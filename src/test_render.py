from pokemon_red_env import PokemonRedEnv
import time

# Usa "SDL2" y frame_skip=1 para ver cada frame en tiempo real
env = PokemonRedEnv("roms/PokemonRed.gb", window="SDL2", frame_skip=1)
obs, info = env.reset()
print("Empezando emulación en modo real-time. Presiona Ctrl+C para detener.")
try:
    while True:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample(), frame_skip=1, render=True)
        env.render()  # Actualiza la ventana
        time.sleep(0.016)  # Aproximadamente 60 fps
        if terminated or truncated:
            break
except KeyboardInterrupt:
    pass
env.close()
print("Emulación finalizada.")
