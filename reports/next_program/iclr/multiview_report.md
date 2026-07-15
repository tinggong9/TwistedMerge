# Genuine multiview coordinate retransport

Execution commit: `74b1d5e9779324615eaf21fe53e7a8f8639190d2`. Five collections used four trained view-specific CNN experts on Princeton ModelNet10 surface samples. The observed 3D sensor frame is explicitly inverted and mapped into each expert frame before rasterization; output mixing alone is not labeled as retransport. Seven graph/view conditions, five calibration resamples, and four 200-draw null families were executed. Dataset archive SHA-256: `9d8679435fc07d1d26f13009878db164a7aa8ea5e7ea3c8880e42794b7307d51`. The complete gate did not pass.
