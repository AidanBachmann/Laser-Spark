# ---------- Import User Files ----------

import Crop as crop
import signalProcessing as SP

# ---------- Flags ----------

to_do = 'crop' # Sets function of main. Options are 'crop' and 'unwrap'.

# ---------- Main ----------

if to_do == 'crop':
    crop.cropAll() # Crop all images
elif to_do == 'unwrap':
    pass
else:
    print('Unknown function. Options are "crop" and "unwrap".')