#################
import os

print("\n=== RUN WITHOUT VIDEO ===")

os.system(
    "python multimodal_data_generator.py "
    "--normal 0 "
    "--cascade 1 "
    "--stressed 0 "
    "--sequence-length 200 "
    "--output-dir debug_no_video"
)

print("\n=== RUN WITH VIDEO ===")

os.system(
    "python multimodal_data_generator.py "
    "--normal 0 "
    "--cascade 1 "
    "--stressed 0 "
    "--sequence-length 200 "
    "--output-dir debug_with_video "
    "--video-path video/wildfire1.mp4"
)
#%#