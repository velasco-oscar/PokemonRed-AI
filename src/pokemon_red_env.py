import gymnasium as gym
from gymnasium import Env, spaces
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import time
import hashlib

class PokemonRedEnv(Env):
    """
    Entorno de PyBoy para Pokémon Rojo basado en la API de Gymnasium.
    Se puede configurar para avanzar múltiples frames por acción (frame skip)
    y renderizar solo el último frame, para acelerar el entrenamiento o visualizar el juego.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, rom_path="roms/PokemonRed.gb", window="null", frame_skip=1):
        super(PokemonRedEnv, self).__init__()
        self.rom_path = rom_path
        self.frame_skip = frame_skip
        # Inicializamos PyBoy en el modo indicado ("null" para headless o "SDL2" para visualización)
        self.pyboy = PyBoy(rom_path, window=window)
        self.pyboy.set_emulation_speed(0)  # Velocidad máxima
        
        # Definir 8 acciones básicas
        self.actions = {
            0: WindowEvent.PRESS_ARROW_UP,
            1: WindowEvent.PRESS_ARROW_DOWN,
            2: WindowEvent.PRESS_ARROW_LEFT,
            3: WindowEvent.PRESS_ARROW_RIGHT,
            4: WindowEvent.PRESS_BUTTON_A,
            5: WindowEvent.PRESS_BUTTON_B,
            6: WindowEvent.PRESS_BUTTON_START,
            7: WindowEvent.PRESS_BUTTON_SELECT
        }
        self.action_space = spaces.Discrete(len(self.actions))
        # Espacio de observación: imagen RGB de 160x144 píxeles
        self.observation_space = spaces.Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8)
        # Para seguimiento de estados nuevos (novelty)
        self.visited_states = set()

    def reset(self, seed=None, options=None):
        if options is None:
            options = {}
        self.pyboy.stop()
        # Reinicia el emulador; aquí usamos "SDL2" para visualización o "null" para entrenamiento headless
        self.pyboy = PyBoy(self.rom_path, window="SDL2")
        self.pyboy.set_emulation_speed(0)
        time.sleep(1)  # Espera mínima para cargar la ROM
        self.visited_states = set()
        return self._get_observation(), {}

    def step(self, action, frame_skip=1, render=False):
        # Convierte la acción a entero si es un arreglo de NumPy
        if isinstance(action, np.ndarray):
            action = int(action.item())
            
        if action not in self.actions:
            raise ValueError(f"Acción no válida: {action}")
    
        # Envía la entrada correspondiente
        self.pyboy.send_input(self.actions[action])
        running = True
        for i in range(frame_skip):
            running = self.pyboy.tick(render=(render and i == frame_skip - 1))
            if not running:
                break
        obs = self._get_observation()
        reward = self._compute_reward(obs)
        terminated = not running   # Terminamos si el tick retorna False
        truncated = False          # No usamos truncado en este ejemplo
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        return self._get_observation()

    def close(self):
        self.pyboy.stop()

    def _get_observation(self):
        # Extrae la imagen actual del emulador y la convierte a un array NumPy.
        image = self.pyboy.screen.image
        obs = np.array(image)
        if len(obs.shape) == 2:  # Si es escala de grises, replicar en 3 canales
            obs = np.stack([obs] * 3, axis=-1)
        elif len(obs.shape) == 3 and obs.shape[-1] == 4:  # Si tiene canal alfa, descartarlo
            obs = obs[..., :3]
        return obs

    def _compute_reward(self, obs):
        """
        Función de recompensa combinada:
         - Novelty: +1 si la observación (imagen) no ha sido vista antes.
         - Penalización por tick: -0.01.
         - (Placeholder) Recompensa por progreso: 0.
        """
        obs_bytes = obs.tobytes()
        obs_hash = hashlib.md5(obs_bytes).hexdigest()
        novelty_reward = 1 if obs_hash not in self.visited_states else 0
        if novelty_reward:
            self.visited_states.add(obs_hash)
        time_penalty = -0.01
        progress_reward = 0
        return novelty_reward + time_penalty + progress_reward

# Bloque de prueba para el entorno (opcional)
if __name__ == "__main__":
    env = PokemonRedEnv("roms/PokemonRed.gb", window="null", frame_skip=100)
    obs, info = env.reset()
    print("Observación inicial:", obs.shape)
    done = False
    tick_count = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        tick_count += env.frame_skip
        print(f"Tick {tick_count} - Recompensa: {reward}")
        done = terminated or truncated
    env.close()
    print("Prueba del entorno finalizada.")
