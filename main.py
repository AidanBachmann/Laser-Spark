# ---------- Imports ----------

import time as TIME
import Crop as CROP
import signalProcessing as SP
import shockVelocity as SV

# ---------- Flags ----------

to_do = 'pass' # Sets function of main. Options are 'crop', 'unwrap', and 'findv'.

# ---------- Main ----------

start = TIME.time()
if to_do == 'crop':
    CROP.cropAll() # Crop all images
elif to_do == 'unwrap':
    #SP.unwrapSingle(shotNumber=21) # 165
    SP.unwrapAll()
elif to_do == 'findv':
    pass
else:
    print('Unknown function. Options are "crop" and "unwrap".')
end = TIME.time()
print(f'\nCode finished in {end - start} seconds.')