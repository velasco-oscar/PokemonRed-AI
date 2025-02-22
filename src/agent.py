import time
from pokemon_red_env import PokemonRedEnv
import numpy as np

def run_random_agent(episodes=5, max_steps=1000):
    """
    Ejecuta un agente aleatorio sobre el entorno durante un número de episodios.
    """
    env = PokemonRedEnv("roms/PokemonRed.gb")
    for episode in range(episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        step = 0
        print(f"Iniciando episodio {episode + 1}")
        while not done and step < max_steps:
            # Elige una acción aleatoria
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1
            # Opcional: imprime algunos detalles cada cierto número de pasos
            if step % 100 == 0:
                print(f"  Paso {step}, recompensa acumulada: {total_reward:.2f}")
        print(f"Episodio {episode + 1} finalizado en {step} pasos, recompensa total: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    run_random_agent()
