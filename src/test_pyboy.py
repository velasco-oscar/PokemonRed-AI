from pyboy import PyBoy
from pyboy.utils import WindowEvent
import time

pyboy = PyBoy("roms/PokemonRed.gb", window="SDL2")
print("Emulador iniciado. Esperando 2 segundos para cargar...")
time.sleep(2)

# Envía la entrada para iniciar (usa el evento correspondiente, por ejemplo, PRESS_BUTTON_START)
pyboy.send_input(WindowEvent.PRESS_BUTTON_START)

tick_count = 0
while pyboy.tick():
    tick_count += 1
    if tick_count % 1000 == 0:
        print("Ticks:", tick_count)
    time.sleep(0.01)


print(f"Emulación finalizada después de {tick_count} ticks. Presiona Enter para salir.")
input()
pyboy.stop()
