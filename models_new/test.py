import os
import sys
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR,'sampling/'))

print(os.path.join(BASE_DIR,'sampling/'))
import tf_sampling
