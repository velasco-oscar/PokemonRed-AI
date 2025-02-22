from pokemon_red_env import PokemonRedEnv
import time

if __name__ == "__main__":
    # Usa "SDL2" para ver la ventana; si deseas entrenar sin visualización, cambia a "null"
    env = PokemonRedEnv(window="SDL2")
    obs = env.reset()
    print("Observación inicial:", obs.shape)
    print("Debug inicial:", env.get_debug_info())
    done = False
    tick_count = 0

    while not done:
        action = env.action_space.sample()  # Acción aleatoria
        obs, reward, done, info = env.step(action)
        tick_count += 1
        print(f"Tick {tick_count} - Recompensa: {reward} - Debug: {env.get_debug_info()}")
        # Delay para procesar los eventos de la ventana
        if env.window != "null":
            time.sleep(0.05)
    env.close()
    print("Prueba del entorno finalizada.")
