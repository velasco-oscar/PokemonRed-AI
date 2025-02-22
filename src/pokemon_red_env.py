import gymnasium as gym
from gymnasium import Env, spaces
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import time
import hashlib
from PIL import Image, ImageDraw, ImageFont  # Para el overlay

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
        self.window = window
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
        
        # Diccionario de etiquetas de acción para overlay
        self.action_labels = {
            0: "ARRIBA",
            1: "ABAJO",
            2: "IZQUIERDA",
            3: "DERECHA",
            4: "A",
            5: "B",
            6: "START",
            7: "SELECT"
        }
        # Espacio de observación: imagen RGB de 160x144 píxeles
        self.observation_space = spaces.Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8)
        # Para seguimiento de estados nuevos (novelty)
        self.visited_states = set()
        # Para almacenar la última acción ejecutada
        self.last_action = None
        self.last_action_text = ""

    def reset(self, seed=None, options=None):
        if options is None:
            options = {}
        self.pyboy.stop()
        # Reinicia el emulador usando el valor almacenado en self.window
        self.pyboy = PyBoy(self.rom_path, window=self.window)
        self.pyboy.set_emulation_speed(0)
        time.sleep(2)  # Tiempo de carga aumentado
        # Envía una pulsación inicial (por ejemplo, START) para iniciar el juego
        self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
        self.visited_states = set()
        self.last_action = None
        self.last_action_text = ""
        return self._get_observation(), {}

    def step(self, action, frame_skip=1, render=False):
        # Convierte la acción a entero si es un arreglo de NumPy
        if isinstance(action, np.ndarray):
            action = int(action.item())
        
        if action not in self.actions:
            raise ValueError(f"Acción no válida: {action}")
    
        # Guarda la acción en formato legible
        self.last_action_text = self.action_labels.get(action, str(action))
        
        # Para incentivar variedad: si la acción actual es diferente de la anterior, se dará un bono
        action_bonus = 0
        if self.last_action is not None:
            if action != self.last_action:
                action_bonus = 0.05  # Bono por cambiar la acción
            else:
                action_bonus = -0.05  # Penalización leve por repetir
        self.last_action = action
        
        print(f"Ejecutando acción: {action} -> {self.actions[action]} ({self.last_action_text})")
        self.pyboy.send_input(self.actions[action])
        running = True
        for i in range(frame_skip):
            running = self.pyboy.tick(render=(render and i == frame_skip - 1))
            if not running:
                break
        obs = self._get_observation()
        base_reward = self._compute_reward(obs)
        # Suma el bono de acción a la recompensa
        reward = base_reward + action_bonus
        terminated = not running
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        # Devuelve la imagen con overlay mostrando la última acción
        return self._get_observation_with_overlay(self.last_action_text)

    def close(self):
        self.pyboy.stop()

    def _get_observation(self):
        # Extrae la imagen actual del emulador y la convierte a un array NumPy.
        image = self.pyboy.screen.image
        obs = np.array(image)
        if len(obs.shape) == 2:  # Si es escala de grises, replicar a 3 canales
            obs = np.stack([obs] * 3, axis=-1)
        elif len(obs.shape) == 3 and obs.shape[-1] == 4:  # Descarta canal alfa
            obs = obs[..., :3]
        return obs

    def _get_observation_with_overlay(self, action_text=""):
        """
        Extrae la imagen actual, dibuja un overlay con el texto de la acción actual,
        y devuelve el array de la imagen resultante.
        """
        image = self.pyboy.screen.image.convert("RGB")
        draw = ImageDraw.Draw(image)
        # Puedes usar una fuente personalizada si la tienes, por defecto se usa la fuente estándar.
        draw.text((10, 10), f"Acción: {action_text}", fill=(255, 0, 0))
        return np.array(image)

    def _compute_reward(self, obs):
        """
        Función de recompensa combinada:
         - Novelty: +0.01 si la observación (imagen) no ha sido vista antes.
         - Penalización por tick: -0.01.
         - Bono por acción: (se aplica en step()).
         - (Placeholder) Recompensa por progreso: 0.
        """
        obs_bytes = obs.tobytes()
        obs_hash = hashlib.md5(obs_bytes).hexdigest()
        novelty_reward = 0.01 if obs_hash not in self.visited_states else 0
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
