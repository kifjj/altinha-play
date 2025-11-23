# altinha-play

This is POC project that applies computer vision model to the sport of Altinha.
Most of the code is written by Cursor (Sonnet 4.5) over a period of 3 rainy days.

The code was fine tuned in Colab and as the free credit finishe I used Kaggle.

1. The first increment: count the number of passes.
2. The second increment: count Head passes and Foot passes.

Some challenges:

- YOLO 11 didn't track properly the ball, I finetunes it with 150 pics annotated in Roboflow.
  As this is a POC, the 150 pics were sampled from a single video with yelllow ball,sandy beach and palms therefore it doesn't work well for different ball colors and environment

- Cursor applied a tracking logic to the ball which didn't work well for Altinha (most ball boxes were lost).
  So I used the best (highest confidence) bounding box directly out of the finetuned model.
  This produced good results.

- the ball hit logic is based on local min (compares the y for the last 3 frames) to detect the hit. Works well.

- the second increment was more challenging. It uses the Yolo Pose to detect the key body parts points and then calculates the closest kbpp to the ball in the detected hit frame.
    - Detecting the closest person is not accurate, very dependent on the video angle, need more work. Perhaps (use the ball size to detect the closes person or tracking logic)
    - initial Cursor logic used the last frame as hit frame, while the correct way is to use the previous local min frame
    - occlusions are a problem, here the tracking logic would help


- In general, while explictly asking Cursor to use Clean Code principle, the result is far from
  perfect. 


Next steps:
- Fine tune with different ball colors and environment
- Create a test set in json format for different videos or even a few frames to track accuracy improvements for a more streamlined development
- Improve accuracy when occlusion occurs
- Improve player detection accuracy (test with different cam angles)
- Consider a machine learning logic that needs to be trained ad-hoc for the specific cases
- Count other body part hits 



