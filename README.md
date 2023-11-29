# AP03MP

Python files for analysing the channel 1 and 9 images - main script (newap03mp.py) uses the cloud filter to calculate channel 1 radiance over ocean and land at local noon, then channel 9 radiance every hour over a 24h period for each of the locations, before calculating an average. (Can first plot channel 9 values to ensure that they follow an approximately sinusoidal variation with the time of day.

**testing.py**
Additional file used for developing the cloud filter to test parameters for the night and day filters, in order to view the images and cloud masks for each hour sequentially.
