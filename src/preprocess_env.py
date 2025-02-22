import gym
from gym.wrappers import GrayScaleObservation, FrameStack, ResizeObservation
from pokemon_red_env import PokemonRedEnv

def main():
    # Crea la instancia base de tu entorno
    # Usamos "null" para modo headless y entrenamiento rápido; si deseas visualizar, cambia a "SDL2"
    env = PokemonRedEnv(window="null")
    
    # Redimensiona la observación a 84x84 píxeles
    env = ResizeObservation(env, shape=84)
    
    # Convierte la observación a escala de grises
    env = GrayScaleObservation(env)
    
    # Apila 4 frames consecutivos para capturar la dinámica del juego
    env = FrameStack(env, num_stack=4)
    
    # Reinicia el entorno y obtiene la primera observación
    obs = env.reset()
    
    # Imprime la forma de la observación preprocesada
    # Debería ser algo similar a (4, 84, 84)
    print("Shape de la observación preprocesada:", obs.shape)
    
    # Ejemplo: Ejecutar algunos pasos y ver recompensas
    done = False
    tick = 0
    while not done and tick < 50:
        action = env.action_space.sample()  # acción aleatoria
        obs, reward, done, info = env.step(action)
        tick += 1
        print(f"Tick {tick} - Recompensa: {reward}")
    
    env.close()

if __name__ == "__main__":
    main()
