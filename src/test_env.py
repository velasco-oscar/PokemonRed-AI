import time
from pokemon_red_env import PokemonRedEnv

if __name__ == "__main__":
    env = PokemonRedEnv()
    obs = env.reset()
    print("Observación inicial:", obs.shape)
    done = False
    tick_count = 0

    # Bucle que continúa hasta que la emulación finalice
    while not done:
        action = env.action_space.sample()  # Acción aleatoria
        obs, reward, done, info = env.step(action)
        tick_count += 1
        print(f"Tick {tick_count} - Recompensa: {reward}")
        time.sleep(0.1)
    
    env.close()
    print("Prueba del entorno finalizada.")
