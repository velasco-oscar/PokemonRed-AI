import gym
from gym import spaces
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import time
import hashlib

class PokemonRedEnv(gym.Env):
    """
    Entorno de PyBoy para Pokémon Rojo basado en la API de OpenAI Gym.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, rom_path="roms/PokemonRed.gb", window="null"):
        super(PokemonRedEnv, self).__init__()
        self.rom_path = rom_path
        self.pyboy = PyBoy(rom_path, window=window)
        
        # Definir espacio de acciones: 8 acciones básicas (arriba, abajo, izquierda, derecha, A, B, Start, Select)
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
        
        # Variable para almacenar estados únicos (por ejemplo, mediante hash de la imagen)
        self.visited_states = set()

    def reset(self):
        self.pyboy.stop()
        self.pyboy = PyBoy(self.rom_path, window="SDL2")
        time.sleep(2)  # Espera a que el juego cargue
        # Reiniciamos el set de estados visitados, pues un nuevo episodio empieza desde cero.
        self.visited_states = set()
        return self._get_observation()

    def step(self, action):
        if action not in self.actions:
            raise ValueError(f"Acción no válida: {action}")
        
        # Envía la entrada correspondiente
        self.pyboy.send_input(self.actions[action])
        
        # Avanza un tick (tick() devuelve True mientras el juego continúa)
        running = self.pyboy.tick()
        
        # Obtén la observación actual (imagen de pantalla)
        obs = self._get_observation()
        
        # Calcula la recompensa basada en si el estado (imagen) es nuevo
        reward = self._compute_reward(obs)
        
        # Define que el episodio termina cuando el juego finaliza (tick() retorna False)
        done = not running
        
        info = {}
        return obs, reward, done, info

    def render(self, mode='human'):
        return self._get_observation()

    def close(self):
        self.pyboy.stop()

    def _get_observation(self):
        # Extrae la imagen actual del emulador
        image = self.pyboy.screen.image
        obs = np.array(image)
        # Si es 2D (escala de grises), lo convertimos a 3 canales replicando la imagen
        if len(obs.shape) == 2:
            obs = np.stack([obs] * 3, axis=-1)
        # Si tiene 4 canales (RGBA), descartamos el canal alfa
        elif len(obs.shape) == 3 and obs.shape[-1] == 4:
            obs = obs[..., :3]
        return obs

    def _compute_reward(self, obs):
        """
        Función de recompensa para premiar la exploración de nuevas áreas.
        Se calcula un hash MD5 de la observación (imagen) y se compara con estados previamente vistos.
        Si es nuevo, se otorga una recompensa positiva.
        """
        # Convertir la observación a bytes
        obs_bytes = obs.tobytes()
        obs_hash = hashlib.md5(obs_bytes).hexdigest()
        if obs_hash not in self.visited_states:
            self.visited_states.add(obs_hash)
            return 1  # Recompensa por explorar un nuevo estado
        return 0

# Ejemplo de prueba (puedes usarlo en test_env.py)
if __name__ == "__main__":
    env = PokemonRedEnv()
    obs = env.reset()
    print("Observación inicial:", obs.shape)
    done = False
    tick_count = 0

    while not done and tick_count < 100:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        tick_count += 1
        print(f"Tick {tick_count} - Recompensa: {reward}")
        time.sleep(0.1)
    env.close()
    print("Prueba del entorno finalizada.")
