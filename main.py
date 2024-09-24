# ---------- Imports ----------

import time as TIME
import Crop as crop
import signalProcessing as SP

# ---------- Flags ----------

to_do = 'pass' # Sets function of main. Options are 'crop' and 'unwrap'.

# ---------- Main ----------

start = TIME.time()
if to_do == 'crop':
    crop.cropAll() # Crop all images
elif to_do == 'unwrap':
    #SP.unwrapSingle(shotNumber=21) # 165
    SP.unwrapAll()
else:
    print('Unknown function. Options are "crop" and "unwrap".')
end = TIME.time()
print(f'\nCode finished in {end - start} seconds.')