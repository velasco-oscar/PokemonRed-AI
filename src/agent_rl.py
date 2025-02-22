import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback
from pokemon_red_env import PokemonRedEnv
import time

# Callback para evaluar y renderizar el entorno cada eval_freq pasos
class RenderEvalCallback(BaseCallback):
    def __init__(self, eval_freq, verbose=0):
        super(RenderEvalCallback, self).__init__(verbose)
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            print("Ejecutando evaluación en modo render...")
            # Creamos un entorno nuevo en modo real-time (ventana visible, 1 frame por acción)
            eval_env = PokemonRedEnv("roms/PokemonRed.gb", window="SDL2", frame_skip=1)
            obs, info = eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action, frame_skip=1, render=True)
                done = terminated or truncated
                eval_env.render()
                time.sleep(0.016)  # Aproximadamente 60 fps
            eval_env.close()
            print("Evaluación finalizada.")
        return True

def main():
    # Selección del modo de ejecución:
    # "1": rápido sin render (alta cantidad de frames por acción)
    # "2": real-time con render (1 frame por acción, para visualización)
    mode = input("Elige modo (1: rápido sin render, 2: real-time con render): ")
    if mode.strip() == "1":
        frame_skip = 100
        render_flag = False
        window = "null"
    else:
        frame_skip = 1
        render_flag = True
        window = "SDL2"

    # Crea el entorno base
    base_env = PokemonRedEnv("roms/PokemonRed.gb", window=window, frame_skip=frame_skip)
    
    # Verifica el entorno base
    check_env(base_env, warn=True)
    
    # Envuelve el entorno en un DummyVecEnv y luego en VecTransposeImage
    env = DummyVecEnv([lambda: base_env])
    env = VecTransposeImage(env)
    
    print("Iniciando entrenamiento...")
    model = DQN("CnnPolicy", env, verbose=1, buffer_size=50000)
    
    # Callback para evaluación en modo render cada 2000 pasos
    callback = RenderEvalCallback(eval_freq=2000)
    model.learn(total_timesteps=10000, callback=callback)
    
    model.save("dqn_pokemon_red")
    print("Modelo guardado.")
    
    # Prueba del agente entrenado en el entorno base (sin vectorizar)
    obs, info = base_env.reset()  # Aquí 'base_env' es la instancia que creaste inicialmente (sin envolver)
    done = False    
    total_ticks = 0
    print("Iniciando prueba del agente entrenado...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = base_env.step(action, frame_skip=1, render=True)
        total_ticks += 1  # Si frame_skip=1, incrementamos de 1 en 1
        print(f"Tick {total_ticks} - Acción: {action} - Recompensa: {reward}")
        done = terminated or truncated
    base_env.close()
    print("Prueba del agente finalizada.")

if __name__ == "__main__":
    main()
