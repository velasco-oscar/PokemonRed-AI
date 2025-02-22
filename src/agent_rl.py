import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from pokemon_red_env import PokemonRedEnv
import time

# Callback para evaluar y renderizar el entorno cada eval_freq pasos con overlay
class RenderEvalCallback(BaseCallback):
    def __init__(self, eval_freq, max_eval_steps=5000, verbose=0):
        super(RenderEvalCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.max_eval_steps = max_eval_steps

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            print("Ejecutando evaluación en modo render...")
            # Creamos un entorno de evaluación en modo real-time (ventana visible, 1 frame por acción)
            eval_env = PokemonRedEnv("roms/PokemonRed.gb", window="SDL2", frame_skip=1)
            obs, info = eval_env.reset()
            done = False
            eval_steps = 0
            while not done and eval_steps < self.max_eval_steps:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action, frame_skip=1, render=True)
                done = terminated or truncated
                # El método render() mostrará la imagen con overlay
                eval_env.render()
                time.sleep(0.016)  # Aproximadamente 60 fps
                eval_steps += 1
            eval_env.close()
            print("Evaluación finalizada tras", eval_steps, "pasos.")
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
    
    # Verifica que el entorno base cumple con la API de Gymnasium
    check_env(base_env, warn=True)
    
    # Envuelve el entorno en un DummyVecEnv y luego en VecTransposeImage (necesario para Stable Baselines3 con imágenes)
    env = DummyVecEnv([lambda: base_env])
    env = VecTransposeImage(env)
    
    print("Iniciando entrenamiento...")

    model_file = "dqn_pokemon_red.zip"
    if os.path.exists(model_file):
        model = DQN.load(model_file, env=env)
        print("Modelo cargado desde", model_file)
    else:
        model = DQN("CnnPolicy", env, verbose=1, buffer_size=50000)
    
    # Callback para guardar checkpoints cada 2000 pasos
    checkpoint_callback = CheckpointCallback(save_freq=2000, save_path='./checkpoints/', name_prefix='dqn_checkpoint')
    # Callback para evaluación en modo render cada 2000 pasos
    render_callback = RenderEvalCallback(eval_freq=2000)
    callbacks = [checkpoint_callback, render_callback]

    # Entrena el modelo (ajusta total_timesteps según lo que necesites)
    model.learn(total_timesteps=10000, callback=callbacks)
    
    # Guarda el modelo final
    model.save("dqn_pokemon_red")
    print("Modelo guardado.")

    # Prueba del agente entrenado en el entorno base (sin vectorizar)
    obs, info = base_env.reset()
    done = False    
    total_ticks = 0
    print("Iniciando prueba del agente entrenado...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = base_env.step(action, frame_skip=1, render=True)
        total_ticks += 1
        print(f"Tick {total_ticks} - Acción: {action} ({base_env.last_action_text}) - Recompensa: {reward}")
        done = terminated or truncated
    base_env.close()
    print("Prueba del agente finalizada.")

if __name__ == "__main__":
    main()
